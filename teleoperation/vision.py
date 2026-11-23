"""Latest-frame worker: body and hand results always share one source frame.

Pipeline per camera frame:
1. Pose Landmarker (VIDEO mode) on the full frame -> shoulders, elbow, wrist.
2. A square window around the selected hand is cropped, resized and passed to
   Hand Landmarker (VIDEO mode), so the fingers get enough pixels. The window
   stays still while the hand moves inside it and only re-centres when the
   hand nears its edge: a window that moved every frame kept shifting the
   hand in the model's input and broke MediaPipe's own frame-to-frame hand
   tracking (frequent re-detection = jumps and dropouts).
3. The hand whose wrist is nearest the selected pose wrist (or the previously
   accepted hand) is used. MediaPipe's hand "Left/Right" labels are not used:
   on an unmirrored camera they are the opposite of the pose model's side.

Each observation also carries the raw per-view 2D landmarks and confidences so
that two cameras can be fused (teleoperation/stereo.py).
"""
from collections import deque
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import sys
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"cv_hand_tracking"))
from track_hand import Frame, ensure_model
from hand_geometry import features, validate_geometry
from teleoperation.arm_tracking import (CHAINS, ArmContinuity, LandmarkFilter, arm_features,
                                        part_status, pose_arrays)

POSE_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{0}/float16/1/pose_landmarker_{0}.task"
POSE_SHA = {"lite": "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
            "full": "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1",
            "heavy": "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"}
HAND_INPUT = 288        # pixels of the square hand crop passed to Hand Landmarker
HAND_EDGE_MARGIN = .05  # hand landmarks may overshoot the image edge by this much (normalized)
VISIBILITY_HOLD = .6    # once the arm is tracked, keep it down to this fraction of min_visibility
HISTORY = 8             # recent observations kept for two-camera time pairing


def ensure_models(variant="full"):
    if variant not in POSE_SHA:
        raise ValueError("Pose model must be lite, full or heavy")
    hand = ensure_model(ROOT.parent/"cv_hand_tracking/models/hand_landmarker.task")
    pose = ROOT/f"models/pose_landmarker_{variant}.task"
    if not pose.exists():
        print(f"Downloading body pose model ({variant}, one time)...", flush=True)
        pose.parent.mkdir(parents=True, exist_ok=True)
        data = urllib.request.urlopen(POSE_BASE.format(variant), timeout=60).read()
        if hashlib.sha256(data).hexdigest() != POSE_SHA[variant]:
            raise ValueError("Body model checksum mismatch")
        temporary = pose.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(pose)
    return pose, hand


class HandWindow:
    """Square crop around the hand that stays still while the hand is well inside it."""

    def __init__(self, min_size=150, recentre=.22, resize=.30):
        self.min_size, self.recentre, self.resize = min_size, recentre, resize
        self.center = self.size = None
        self.missed = 0

    def reset(self):
        self.center = self.size = None
        self.missed = 0

    def update(self, shape, wrist=None, elbow=None, hand=None):
        height, width = shape[:2]
        forearm = None if wrist is None or elbow is None else float(np.linalg.norm(wrist-elbow))
        if hand is not None:
            low, high = hand.min(axis=0), hand.max(axis=0)
            center, size = (low+high)/2, float(np.max(high-low))*2.3
            if forearm:
                size = max(size, forearm*1.1)
            self.missed = 0
        elif wrist is not None and elbow is not None:
            # Fingers extend beyond the wrist roughly along the forearm. While the
            # hand stays lost, search a progressively larger window.
            self.missed += 1
            grow = 1.+min(self.missed, 6)*.12
            center, size = wrist+.45*(wrist-elbow), max(forearm*1.9*grow, self.min_size)
        elif self.center is not None and self.missed < 8:
            self.missed += 1
            center, size = self.center, self.size*1.08
        else:
            self.reset()
            return None
        size = float(np.clip(size, self.min_size, min(width, height)))
        if (self.center is None or np.linalg.norm(center-self.center) > self.recentre*self.size
                or abs(size/self.size-1) > self.resize or self.missed > 0):
            # Re-centre only when needed; between re-centring events the crop is fixed.
            self.center, self.size = np.asarray(center, float), size
        side = int(round(self.size))
        left = int(np.clip(round(self.center[0]-side/2), 0, width-side))
        top = int(np.clip(round(self.center[1]-side/2), 0, height-side))
        return left, top, left+side, top+side


@dataclass
class Observation:
    frame: object
    completed_ns: int
    arm: dict | None
    hand: dict | None
    hand_image: np.ndarray | None
    pose_image: np.ndarray | None
    reason: str
    roi: tuple | None = None
    parts: list = field(default_factory=list)
    arm_reason: str = ""
    hand_reason: str = ""
    hand_label: str | None = None
    timing_ms: dict = field(default_factory=dict)
    # Raw per-view data for two-camera fusion.
    pose_confidence: np.ndarray | None = None
    pose_world: np.ndarray | None = None
    hand_world: np.ndarray | None = None   # MediaPipe hand world landmarks (camera axes), only for an accepted hand
    image_size: tuple = (0, 0)
    # Two-camera composite only: the per-view observations and fusion details.
    views: tuple = ()
    fusion: dict = field(default_factory=dict)


def select_arm_hand(result, wrist_px, previous_px, size, max_gap_px):
    """Return (image_landmarks, world_landmarks, label, reason) for the hand nearest the selected wrist.

    A hand is accepted when it is near the pose wrist or near the previously
    accepted hand: the pose wrist estimate is often off by a few centimetres
    when the hand is in front of the body.
    """
    best, best_gap, labels = None, np.inf, []
    references = [r for r in (wrist_px, previous_px) if r is not None]
    for index, landmarks in enumerate(result.hand_landmarks):
        image = np.array([[p.x, p.y, p.z] for p in landmarks])
        world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[index]])
        label = result.handedness[index][0].category_name if result.handedness and result.handedness[index] else "?"
        labels.append(label)
        gap = 0. if not references else min(float(np.linalg.norm(image[0, :2]*size-r)) for r in references)
        if gap < best_gap:
            best, best_gap = (image, world, label), gap
    if best is None:
        return None, None, None, "Hand not found near the selected wrist: open your hand toward the camera"
    if references and best_gap > max_gap_px:
        return best[0], None, best[2], "The detected hand does not belong to the selected arm"
    return best[0], best[1], best[2], ""


def check_hand(image, world):
    """validate_geometry, tolerating a small overshoot of the image edge (fingertips at the border)."""
    if image is not None and np.isfinite(image).all():
        outside = np.maximum(-image[:, :2], image[:, :2]-1).max()
        if outside > HAND_EDGE_MARGIN:
            return "Hand is partly outside the camera image"
        image = image.copy()
        image[:, :2] = np.clip(image[:, :2], 0, 1)
    return validate_geometry(image, world)


class WholeArmVision:
    def __init__(self, camera, side="right", mirrored=False, min_visibility=.5, pose_model="full"):
        self.camera, self.side, self.mirrored, self.min_visibility = camera, side, mirrored, min_visibility
        self.latest = None
        self.recent = deque(maxlen=HISTORY)
        self.error = None
        self.lock = threading.Lock()
        self.halt = threading.Event()
        self.paths = ensure_models(pose_model)
        self.thread = threading.Thread(target=self._work, daemon=True, name="synchronized-arm-hand")
        self.thread.start()

    def _work(self):
        arm_filter = LandmarkFilter(1.0, 5.0, reset_gap_s=.5)
        hand_filter = LandmarkFilter(2.5, 10.0, reset_gap_s=.3)
        continuity = ArmContinuity()
        window = HandWindow()
        self._last_hand_px = None
        self._arm_tracked = False
        chain = CHAINS[self.side]
        try:
            base = mp.tasks.BaseOptions
            vision = mp.tasks.vision
            # Lower tracking thresholds keep MediaPipe tracking through brief
            # uncertainty instead of re-detecting (re-detection causes jumps).
            pose_options = vision.PoseLandmarkerOptions(
                base_options=base(model_asset_path=str(self.paths[0])), running_mode=vision.RunningMode.VIDEO, num_poses=1,
                min_pose_detection_confidence=.5, min_pose_presence_confidence=.4, min_tracking_confidence=.4)
            hand_options = vision.HandLandmarkerOptions(
                base_options=base(model_asset_path=str(self.paths[1])), running_mode=vision.RunningMode.VIDEO, num_hands=2,
                min_hand_detection_confidence=.4, min_hand_presence_confidence=.4, min_tracking_confidence=.35)
            with vision.PoseLandmarker.create_from_options(pose_options) as pose, \
                    vision.HandLandmarker.create_from_options(hand_options) as hand:
                last_frame, last_ts = -1, -1
                while not self.halt.is_set():
                    raw = self.camera.get()
                    if raw is None or raw.index == last_frame:
                        self.halt.wait(.003)
                        continue
                    last_frame = raw.index
                    timestamp = max(last_ts+1, raw.received_ns//1_000_000)
                    last_ts = timestamp
                    image = cv2.flip(raw.image, 1) if self.mirrored else raw.image
                    frame = Frame(raw.index, image, raw.received_ns, raw.read_ms)
                    observation = self._process(frame, timestamp, pose, hand, arm_filter, hand_filter, continuity, window, chain)
                    with self.lock:
                        self.latest = observation
                        self.recent.append(observation)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"

    def _process(self, frame, timestamp, pose, hand, arm_filter, hand_filter, continuity, window, chain):
        image = frame.image
        height, width = image.shape[:2]
        size = np.array([width, height], float)
        source_s = frame.received_ns/1e9
        started = time.perf_counter()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose_result = pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
        pose_ms = (time.perf_counter()-started)*1000
        arm = hand_measurement = hand_image = pose_image = label = None
        confidence = world = hand_world_raw = None
        arm_reason = hand_reason = ""
        parts = part_status(None, None, self.side)
        wrist_px = elbow_px = None
        # Hysteresis: a tracked arm is kept down to a lower visibility.
        threshold = self.min_visibility*(VISIBILITY_HOLD if self._arm_tracked else 1.)
        try:
            pose_image, world, confidence = pose_arrays(pose_result)
            parts = part_status(pose_image, confidence, self.side, threshold)
            _, elbow_id, wrist_id = chain
            if confidence[wrist_id] > .2:
                wrist_px, elbow_px = pose_image[wrist_id, :2]*size, pose_image[elbow_id, :2]*size
            ids = [11, 12, chain[1], chain[2]]
            filtered = world.copy()
            filtered[ids] = arm_filter.update(world[ids], source_s)
            arm = arm_features(pose_image, filtered, confidence, self.side, threshold)
            continuity.check(arm, source_s)
        except ValueError as error:
            arm, arm_reason = None, str(error)
        self._arm_tracked = arm is not None
        # Hand: crop around the hand (or the predicted hand from the forearm).
        started = time.perf_counter()
        previous = self._last_hand_px
        roi = window.update(image.shape, wrist_px, elbow_px, previous)
        if roi is None:
            hand_reason = "Waiting for the selected arm's wrist to locate the hand"
        else:
            x1, y1, x2, y2 = roi
            crop = cv2.resize(rgb[y1:y2, x1:x2], (HAND_INPUT, HAND_INPUT), interpolation=cv2.INTER_AREA if x2-x1 > HAND_INPUT else cv2.INTER_LINEAR)
            result = hand.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)), timestamp)
            # Map crop-normalized landmarks back to full-image normalized coordinates.
            for landmarks in result.hand_landmarks:
                for p in landmarks:
                    p.x = (p.x*(x2-x1)+x1)/width
                    p.y = (p.y*(y2-y1)+y1)/height
                    p.z = p.z*(x2-x1)/width
            forearm_px = None if wrist_px is None else float(np.linalg.norm(wrist_px-elbow_px))
            max_gap = max(80., 1.0*forearm_px) if forearm_px else 110.
            hand_image, hand_world, label, hand_reason = select_arm_hand(result, wrist_px, None if previous is None else previous[0], size, max_gap)
            if hand_world is not None:
                error = check_hand(hand_image, hand_world)
                if error:
                    hand_reason = error
                else:
                    try:
                        hand_measurement = features(hand_filter.update(hand_world, source_s))
                        hand_world_raw = hand_world
                    except ValueError as failure:
                        hand_reason = str(failure)
        # Only an accepted hand steers the crop window and the next association.
        self._last_hand_px = None if hand_measurement is None else hand_image[:, :2]*size
        hand_ms = (time.perf_counter()-started)*1000
        if hand_measurement is None:
            hand_filter.reset()
        if arm is not None:
            arm["side"] = self.side.title()
        reason = "Synchronized shoulder / elbow / wrist / fingers" if arm and hand_measurement else (arm_reason or hand_reason)
        return Observation(frame, time.perf_counter_ns(), arm, hand_measurement, hand_image, pose_image, reason, roi,
                           parts, arm_reason, hand_reason, label, {"pose": pose_ms, "hand": hand_ms},
                           confidence, world, hand_world_raw, (width, height))

    _last_hand_px = None
    _arm_tracked = False

    def get(self):
        with self.lock:
            return self.latest

    def history(self):
        with self.lock:
            return list(self.recent)

    def close(self):
        self.halt.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Inference worker did not stop; restart the program before changing cameras")
