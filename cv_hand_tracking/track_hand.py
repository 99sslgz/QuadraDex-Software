#!/usr/bin/env python3
"""QuadraDex camera hand acquisition: preview, calibrate, log and stream targets.

Run with --help. This S2 prototype emits geometric observations for S3, not
LEAP motor commands. A downstream controller must enforce its own watchdog.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import queue
import socket
import sys
import threading
import time
import textwrap
import urllib.request
import uuid

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ImportError as error:
    raise SystemExit("Missing dependency. Install with: python -m pip install -r requirements.txt") from error

from hand_geometry import (Calibration, CONNECTIONS, FINGERS, OneEuro, StreamGate,
                           features, make_calibration, validate_geometry)
from calibration_ui import CalibrationCapture, finger_ranges
from camera_picker import CameraChoice, choose_camera, discover_cameras

ROOT = Path(__file__).resolve().parent
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
WINDOW = "QuadraDex | CV hand tracking"
CLOCK_ID = "host_perf_counter_ns"


def ensure_model(path: Path) -> Path:
    if not path.is_file():
        print("Downloading Google's Hand Landmarker model (one time)...", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".download")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=30) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if temporary.stat().st_size < 1_000_000:
                raise ValueError("Downloaded model is unexpectedly small")
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != MODEL_SHA256:
                raise ValueError("Model checksum does not match the pinned Google model")
            temporary.replace(path)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Model download failed: {error}. Download {MODEL_URL} to {path}") from error
    return path


@dataclass
class Frame:
    index: int
    image: np.ndarray
    received_ns: int
    read_ms: float


class LatestCamera:
    """Read continuously into one slot so inference cannot build a frame queue."""

    def __init__(self, index: int, width: int, height: int, fps: float, backend: str):
        api = {"auto": cv2.CAP_ANY, "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}[backend]
        self.capture = cv2.VideoCapture(index, api)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open camera {index}. Close other camera apps or try --camera 1 / --backend dshow.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Backend may ignore this request.
        print(f"Camera reports {self.capture.get(cv2.CAP_PROP_FRAME_WIDTH):.0f} x "
              f"{self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} at "
              f"{self.capture.get(cv2.CAP_PROP_FPS):.1f} FPS (requested settings may be ignored)", flush=True)
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.halt = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True, name="camera")
        self.thread.start()

    def _read(self):
        index, failures = 0, 0
        try:
            while not self.halt.is_set():
                before = time.perf_counter_ns()
                ok, image = self.capture.read()
                after = time.perf_counter_ns()
                if not ok:
                    failures += 1
                    if failures >= 10:
                        self.error = "Camera stopped returning frames"
                        return
                    self.halt.wait(.02)
                    continue
                failures = 0
                index += 1
                with self.lock:
                    self.latest = Frame(index, image, after, (after - before) / 1e6)
        except Exception as error:
            self.error = str(error)
        finally:
            self.capture.release()

    def get(self):
        with self.lock:
            return self.latest

    def close(self):
        self.halt.set()
        self.thread.join(timeout=1)
        # The reader releases capture in its own thread, including on read failure.


class AsyncHands:
    def __init__(self, model: Path, confidence: float):
        self.lock = threading.Lock()
        self.inflight = {}
        self.latest = None
        self.last_timestamp = -1
        self.error = None
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=confidence,
            result_callback=self._callback)
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _callback(self, result, output_image, timestamp_ms):
        try:
            with self.lock:
                frame = self.inflight.pop(timestamp_ms, None)
                if frame is not None:
                    self.latest = (result, frame, time.perf_counter_ns())
        except Exception as error:
            self.error = str(error)

    def submit(self, frame: Frame):
        timestamp = frame.received_ns // 1_000_000
        if timestamp <= self.last_timestamp:
            return
        self.last_timestamp = timestamp
        with self.lock:
            self.inflight[timestamp] = frame
            # MediaPipe may skip inputs while busy; retain only bounded metadata.
            while len(self.inflight) > 16:
                del self.inflight[next(iter(self.inflight))]
        rgb = cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)
        self.detector.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)

    def get(self):
        with self.lock:
            return self.latest

    def close(self):
        self.detector.close()


class Output:
    """Nonblocking UDP plus a bounded background JSONL writer."""

    def __init__(self, destination, log_path):
        self.socket = None
        self.destination = destination
        self.dropped_udp = self.dropped_logs = 0
        self.log_error = None
        self.pending = queue.Queue(maxsize=256)
        self.halt = threading.Event()
        self.writer = None
        self.logfile = None
        if destination:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setblocking(False)
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.logfile = log_path.open("x", encoding="utf-8")
            self.writer = threading.Thread(target=self._write, daemon=True, name="jsonl-logger")
            self.writer.start()

    def _write(self):
        try:
            while not self.halt.is_set() or not self.pending.empty():
                try:
                    line = self.pending.get(timeout=.1)
                except queue.Empty:
                    continue
                self.logfile.write(line + "\n")
            self.logfile.flush()
        except Exception as error:
            self.log_error = str(error)
        finally:
            self.logfile.close()

    def emit(self, packet):
        line = json.dumps(packet, separators=(",", ":"), allow_nan=False)
        if self.socket:
            try:
                self.socket.sendto(line.encode("utf-8"), self.destination)
            except OSError:
                self.dropped_udp += 1
        if self.writer:
            try:
                self.pending.put_nowait(line)
            except queue.Full:
                self.dropped_logs += 1

    def close(self):
        self.halt.set()
        if self.writer:
            self.writer.join(timeout=2)
        if self.socket:
            self.socket.close()


def select_hand(result, expected: str | None, swap: bool, min_score: float):
    candidates, labels = [], []
    for index, categories in enumerate(result.handedness):
        if not categories:
            continue
        top = max(categories, key=lambda category: category.score)
        label = top.category_name
        if swap:
            label = {"Left": "Right", "Right": "Left"}.get(label, label)
        labels.append(f"{label} {top.score:.2f}")
        if (expected is None or label == expected) and top.score >= min_score:
            candidates.append((index, float(top.score), label))
    if len(candidates) != 1:
        if len(candidates) > 1:
            return None, "Multiple hands: show only the hand to control"
        return None, "Camera sees " + (", ".join(labels) or "no hand") + (f"; selected: {expected}" if expected else "")
    index, score, label = candidates[0]
    image = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[index]])
    world = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[index]])
    error = validate_geometry(image, world)
    # Preserve the observed skeleton for diagnostics even if geometry is rejected.
    return (image, world, score, label), error


def preview_landmarks(result):
    if len(result.hand_landmarks) != 1:
        return None
    points = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])
    return points if points.shape == (21, 3) and np.isfinite(points).all() else None


BUTTONS = [("o", "1. Open", 18, 112), ("c", "2. Closed", 122, 224),
           (" ", "Start/Stop", 234, 362), ("v", "Camera", 372, 472), ("h", "Hand", 482, 562)]


def draw(frame, image_points, measurement, state, reason, metrics, note, mirror, ui=None):
    ui = ui or {}
    canvas = frame.image.copy()
    height, width = canvas.shape[:2]
    if mirror:
        canvas = cv2.flip(canvas, 1)
    if image_points is not None:
        points = image_points[:, :2].copy()
        if mirror:
            points[:, 0] = 1 - points[:, 0]
        points = np.rint(points * [width, height]).astype(int)
        skeleton_color = (95, 225, 170) if ui.get("tracking_valid", True) else (80, 185, 255)
        for a, b in CONNECTIONS:
            cv2.line(canvas, tuple(points[a]), tuple(points[b]), skeleton_color, 2, cv2.LINE_AA)
        for index, point in enumerate(points):
            cv2.circle(canvas, tuple(point), 4, (70, 190, 255), -1, cv2.LINE_AA)
            cv2.putText(canvas, str(index), tuple(point + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
    panel = np.full((max(height, 820), 580, 3), (29, 25, 23), dtype=np.uint8)
    color = (100, 230, 160) if state == "STREAMING" else (90, 195, 255)

    def label(text, y, scale=.52, ink=(230, 230, 230)):
        cv2.putText(panel, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, scale, ink, 1, cv2.LINE_AA)

    label(ui.get("title", "QUADRADEX / HAND TRACKING"), 30, .72)
    label(ui.get("camera_name", "Camera")[:58] + "  |  V: choose camera", 58, .46)
    label(state.replace("_", " "), 96, .72, color)
    for i, line in enumerate(textwrap.wrap(reason, width=72)[:2]):
        label(line, 124+i*20, .46)
    label(f"Hand: {ui.get('hand_label', 'Auto')}  |  H: change hand  |  M: flip labels", 172, .46)
    age = metrics['result_age_ms']
    label(f"Fresh measurements: {metrics['valid_result_hz']:.1f}/s   |   Age: {age:.0f} ms" if age is not None else "Waiting for hand measurements...", 200, .50)
    cv2.rectangle(panel, (12, 221), (566, 425), (47, 44, 40), -1)
    progress = ui.get("calibration_progress", {})
    stage = progress.get("stage")
    heading = ("STEP 1 OF 2: OPEN HAND" if stage == "open" else "STEP 2 OF 2: CLOSED HAND") if stage else (
        "CALIBRATION COMPLETE" if ui.get("calibrated") else "STEP 2 OF 2: CLOSE YOUR HAND" if ui.get("opened") else "STEP 1 OF 2: OPEN YOUR HAND")
    label(heading, 250, .64, (115, 225, 170))
    instruction = progress.get("message", note) if stage else note
    for i, line in enumerate(textwrap.wrap(instruction, width=69)[:3]):
        label(line, 280+i*22, .48, (120, 205, 255))
    if stage:
        fraction = progress.get("progress", 0.)
        cv2.rectangle(panel, (24, 346), (550, 365), (75, 71, 65), -1)
        cv2.rectangle(panel, (24, 346), (24+round(526*fraction), 365), (100, 205, 155), -1)
        label(f"{round(100*fraction)}%  |  {progress.get('accepted_samples', 0)} good samples  |  {progress.get('phase', '').upper()}", 390, .52)
    else:
        label("Open pose: " + ("SAVED" if ui.get("opened") or ui.get("calibrated") else "waiting for O"), 350, .52)
        label("Closed pose: " + ("SAVED" if ui.get("calibrated") else "waiting for C"), 376, .52)
    ranges = ui.get("finger_ranges", {})
    if ranges and not ui.get("calibrated"):
        label("Travel: " + "  ".join(f"{finger} {'OK' if value['ready'] else 'bend more'}" for finger, value in ranges.items()), 414, .41)
    label("Finger bend estimates / raw curl", 454, .52)
    y = 482
    for name in FINGERS:
        if measurement:
            joints = measurement["joints_deg"][name]
            values = "/".join(f"{joints[key]:.0f}" for key in (("mcp", "ip") if name == "thumb" else ("mcp", "pip", "dip")))
            label(f"{name:6s} {values:12s} curl {measurement['curl_raw'][name]:.2f}", y)
            cv2.rectangle(panel, (18, y + 8), (550, y + 15), (60, 60, 60), -1)
            cv2.rectangle(panel, (18, y + 8), (18 + int(532 * measurement["curl_raw"][name]), y + 15), (100, 200, 150), -1)
        else:
            label(f"{name:6s} --", y)
        y += 35
    if measurement:
        label(f"Thumb-index gap: {measurement['pinch_distance_palm_widths']['index']:.2f} palm widths", 673)
    label("O / C: capture poses   Space: start/stop stream", 708, .49)
    label("S: stop   R: reset calibration   Q/Esc: quit", 733, .49)
    for key, title, left, right in BUTTONS:
        cv2.rectangle(panel, (left, 755), (right, 796), (70, 78, 68), -1)
        cv2.putText(panel, title, (left+6, 781), cv2.FONT_HERSHEY_SIMPLEX, .44, (240, 240, 240), 1, cv2.LINE_AA)
    padded = np.zeros((panel.shape[0], width, 3), dtype=np.uint8)
    padded[:height] = canvas
    return np.hstack((padded, panel))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=None, help="Skip the camera picker and use this camera index")
    parser.add_argument("--list-cameras", action="store_true", help="List camera names/indices and exit")
    parser.add_argument("--choose-camera", action="store_true", help="Open the picker even when --camera was provided")
    parser.add_argument("--hand", choices=("auto", "right", "left"), default="auto", help="Auto selects the sole visible hand, then locks identity during calibration")
    parser.add_argument("--swap-handedness", action="store_true", help="Invert model labels if your camera provides mirrored input")
    parser.add_argument("--backend", choices=("auto", "dshow", "msmf"), default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=float, default=30)
    parser.add_argument("--output-hz", type=float, default=30, help="Observation/heartbeat cadence; downstream robot loop is separate")
    parser.add_argument("--max-age-ms", type=float, default=100, help="Prototype host-receipt age limit; not a formal project threshold")
    parser.add_argument("--confidence", type=float, default=.5, help="Model internal detection/presence/tracking thresholds")
    parser.add_argument("--handedness-score", type=float, default=.6, help="Hand identity threshold, not a positional quality score")
    parser.add_argument("--min-cutoff", type=float, default=2, help="One Euro smoothing minimum cutoff in Hz")
    parser.add_argument("--beta", type=float, default=12, help="One Euro speed response, in inverse model metres")
    parser.add_argument("--no-smoothing", action="store_true")
    parser.add_argument("--no-mirror", action="store_true", help="Disable display mirror; inference input is never mirrored by this script")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "hand_landmarker.task")
    parser.add_argument("--download-model", action="store_true", help="Fetch model and exit without opening the camera")
    parser.add_argument("--calibration", type=Path, help="Load this profile; O/C also saves completed calibration here")
    parser.add_argument("--udp", metavar="HOST:PORT", help="Optional JSON UDP destination, e.g. 127.0.0.1:5005")
    parser.add_argument("--log", type=Path, help="New JSONL log path; existing files are never overwritten")
    parser.add_argument("--headless", action="store_true", help="Diagnostic observations only; no keyboard calibration or stream start")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    args = parser.parse_args(argv)
    if not (0 < args.confidence <= 1 and 0 < args.handedness_score <= 1):
        parser.error("Confidence thresholds must be in (0, 1]")
    numbers = [args.width, args.height, args.camera_fps, args.output_hz, args.max_age_ms, args.min_cutoff]
    if any(not np.isfinite(number) or number <= 0 for number in numbers) or not np.isfinite(args.beta) or args.beta < 0:
        parser.error("Dimensions, rates, age and cutoff must be positive and finite; beta must be nonnegative")
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("Duration must be positive and finite")
    if args.camera is not None and args.camera < 0:
        parser.error("Camera index must be nonnegative")
    if args.headless and args.choose_camera:
        parser.error("The camera picker needs a display; omit --headless")
    if args.udp:
        try:
            host, port = args.udp.rsplit(":", 1)
            port = int(port)
            if not 1 <= port <= 65535:
                raise ValueError("Port out of range")
            args.udp = (socket.gethostbyname(host), port)
        except (ValueError, OSError):
            parser.error("UDP destination must be a resolvable IPv4 HOST:PORT")
    return args


def inhibit_packet(packet, sequence, reason="Producer stopped"):
    return {**packet, "sequence": sequence, "state": "STOPPED", "new_result": False,
            "published_monotonic_ns": time.perf_counter_ns(), "tracking_valid": False,
            "reason": reason, "measurement": None, "teleop_targets": None,
            "landmarks_image_normalized": None, "landmarks_world_model_m_raw": None,
            "landmarks_world_model_m_filtered": None}


def run(args):
    if args.list_cameras:
        for device in discover_cameras(args.backend):
            print(f"{device.index}: {device.name} (connection: {device.backend})")
        return 0
    model = ensure_model(args.model.resolve())
    if args.download_model:
        print(f"Model ready: {model}\nSHA256: {hashlib.sha256(model.read_bytes()).hexdigest()}")
        return 0
    hand_mode = args.hand
    expected = None if hand_mode == "auto" else hand_mode.title()
    calibration = None
    if args.calibration and args.calibration.exists():
        calibration = Calibration.load(args.calibration, expected)
        expected = calibration.handedness
    gate, capture = StreamGate(), CalibrationCapture()
    output = Output(args.udp, args.log)
    camera = detector = None
    session, session_wall_ns = str(uuid.uuid4()), time.time_ns()
    started_ns = time.perf_counter_ns()
    good_times, inference_times, ages = deque(), deque(maxlen=100000), deque(maxlen=100000)
    frame_id = processed_id = seq = 0
    last_published_frame = None
    measurement = image_points = preview_points = preview_measurement = None
    world_raw = world_filtered = source_frame = score = None
    filter_ = OneEuro(args.min_cutoff, args.beta)
    reason = "Show one hand to begin automatic hand selection" if expected is None else f"Show your {expected.lower()} hand"
    note = "Open your hand and press O, or click 1. Open"
    if calibration is not None:
        note = "Saved calibration loaded. Show your hand, then press Space."
    opened = None
    last_packet = None
    last_console = 0.
    valid = False
    exit_code = 0
    click_keys = []
    metrics = {"valid_result_hz": 0., "result_age_ms": None, "inference_ms": None}
    choice = CameraChoice(args.camera if args.camera is not None else 0, args.backend,
                          f"Camera {args.camera if args.camera is not None else 0}")

    def clicked(event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        camera_width = source_frame.image.shape[1] if source_frame else (camera.get().image.shape[1] if camera and camera.get() else args.width)
        for key, title, left, right in BUTTONS:
            if left <= x-camera_width <= right and 755 <= y <= 796:
                click_keys.append(ord(key))

    try:
        if not args.headless and (args.camera is None or args.choose_camera):
            picked = choose_camera(LatestCamera, args)
            if picked is None:
                return 0
            choice, camera = picked
        else:
            camera = LatestCamera(choice.index, args.width, args.height, args.camera_fps, choice.backend)
        detector = AsyncHands(model, args.confidence)
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, clicked)
        started_ns = time.perf_counter_ns()
        next_tick = next_publish = time.perf_counter()
        print(f"Camera: {choice.name}. O=open, C=closed, Space=start/stop, V=cameras, H=hand, M=flip labels, Q=quit.", flush=True)
        while True:
            now = time.perf_counter()
            if args.duration and now - started_ns / 1e9 >= args.duration:
                break
            frame = camera.get()
            if camera.error or detector.error:
                raise RuntimeError(camera.error or detector.error)
            if frame and frame.index != frame_id:
                detector.submit(frame)
                frame_id = frame.index
            completed = detector.get()
            new_result = bool(completed and completed[1].index != processed_id)
            if new_result:
                result, source_frame, finished_ns = completed
                processed_id = source_frame.index
                selection_expected = expected if (hand_mode != "auto" or calibration is not None or opened is not None or capture.active) else None
                selected, rejection = select_hand(result, selection_expected, args.swap_handedness, args.handedness_score)
                measurement = image_points = world_raw = world_filtered = score = preview_measurement = None
                preview_points = preview_landmarks(result)
                if selected is not None:
                    image_points, world_raw, score, label = selected
                    preview_points = image_points
                    if rejection is None:
                        if expected != label:
                            filter_ = OneEuro(args.min_cutoff, args.beta)
                            expected = label  # Calibration binds the selected hand identity.
                        age = (time.perf_counter_ns() - source_frame.received_ns) / 1e6
                        if age <= 250:
                            try:
                                if filter_.time is not None and source_frame.received_ns / 1e9 - filter_.time > .25:
                                    filter_ = OneEuro(args.min_cutoff, args.beta)
                                world_filtered = world_raw.copy() if args.no_smoothing else filter_.update(world_raw, source_frame.received_ns / 1e9)
                                preview_measurement = features(world_filtered)
                                measurement = preview_measurement
                                if calibration and abs(measurement["palm_width_model_m"] / calibration.palm_width_model_m - 1) > .35:
                                    rejection, measurement = "Palm size differs from calibration; press R to recalibrate", None
                                elif age <= args.max_age_ms:
                                    good_times.append(finished_ns / 1e9)
                                    inference_times.append((finished_ns - source_frame.received_ns) / 1e6)
                            except ValueError as error:
                                rejection = str(error)
                        else:
                            rejection = "Inference is delayed; check the camera frame rate"
                if rejection:
                    reason = rejection
                    filter_ = OneEuro(args.min_cutoff, args.beta)
                else:
                    reason = f"Tracking {expected.lower()} hand"
            current_ns = time.perf_counter_ns()
            age_ms = (current_ns - source_frame.received_ns) / 1e6 if source_frame else None
            valid = measurement is not None and age_ms is not None and age_ms <= args.max_age_ms
            if measurement is not None and not valid:
                reason = "Preview only: measurement too old for targets"
            elif valid:
                reason = f"Tracking {expected.lower()} hand"
            if frame is None and now - started_ns / 1e9 > 5:
                raise RuntimeError("Camera opened but no frames arrived")
            gate.observe(valid, now)
            completed_pose = capture.update(now, measurement if valid else None, new_result,
                                            source_frame.received_ns / 1e9 if source_frame else None, reason)
            if completed_pose:
                stage, pose = completed_pose
                try:
                    if stage == "open":
                        opened = pose
                        note = "Open pose saved. Close your fingers and thumb, then press C"
                    else:
                        profile = make_calibration(expected, opened, pose)
                        path = args.calibration or ROOT / f"calibration_{expected.lower()}.json"
                        profile.save(path)
                        calibration = profile
                        note = "Calibration saved. Hold a visible pose and press Space to stream"
                except (ValueError, OSError) as error:
                    note = f"Could not save {stage} pose: {error}. Press {'C' if stage == 'closed' else 'O'} to retry."
                    capture.message, capture.phase = note, "error"
            elif capture.phase in ("cancelled", "error"):
                note = capture.message
            while good_times and good_times[0] < now - 2.:
                good_times.popleft()
            window = min(2., max(.001, now - started_ns / 1e9))
            metrics = {"valid_result_hz": len(good_times) / window, "result_age_ms": age_ms,
                       "inference_ms": (completed[2] - source_frame.received_ns) / 1e6 if completed and source_frame else None}
            state = gate.state(calibration is not None, capture.active)
            key = -1
            if not args.headless and frame:
                # Preview freshness is independent of the stricter teleoperation gate.
                preview_ok = source_frame is not None and age_ms is not None and age_ms <= 250
                display_frame = source_frame if preview_ok else frame
                progress = capture.status(now)
                hand_label = f"Auto -> {expected or 'show one hand'}" if hand_mode == "auto" else hand_mode.title()
                ui = {"camera_name": choice.name, "hand_label": hand_label, "tracking_valid": valid,
                      "calibration_progress": progress, "opened": opened is not None,
                      "calibrated": calibration is not None,
                      "finger_ranges": finger_ranges(opened, preview_measurement) if preview_measurement else {}}
                cv2.imshow(WINDOW, draw(display_frame, preview_points if preview_ok else None,
                                        preview_measurement if preview_ok else None, state, reason, metrics,
                                        note, not args.no_mirror, ui))
                key = cv2.waitKey(1) & 0xFF
                if click_keys:
                    key = click_keys.pop(0)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                gate.stop()
                capture.cancel("Stopped. Press O or C to capture a pose, or Space to start the stream.")
                note = capture.message
            elif key in (ord("r"), ord("h"), ord("m")):
                gate = StreamGate()
                calibration = opened = None
                capture = CalibrationCapture()
                filter_ = OneEuro(args.min_cutoff, args.beta)
                if key == ord("h"):
                    hand_mode = {"auto": "right", "right": "left", "left": "auto"}[hand_mode]
                if key == ord("m"):
                    args.swap_handedness = not args.swap_handedness
                expected = None if hand_mode == "auto" else hand_mode.title()
                measurement = None
                reason = "Hand selection reset; show the selected hand"
                note = "Calibration cleared. Open your hand and press O."
            elif key == ord("o"):
                gate.stop()
                calibration, opened = None, None
                capture.start("open", now)
            elif key == ord("c"):
                if opened is not None:
                    gate.stop()
                    calibration = None
                    capture.start("closed", now)
                else:
                    note = "Capture the OPEN pose first: open your hand and press O"
                    capture.message, capture.phase = note, "error"
            elif key == ord(" ") and not capture.active:
                capture.phase = "idle"
                if gate.active:
                    gate.stop()
                    note = "Target stream stopped. Press Space to start again."
                elif gate.start(calibration is not None, valid, now):
                    note = "Target stream running. Press S or Space to stop."
                else:
                    note = "Need both calibration poses and 0.5 seconds of fresh tracking."
            elif key == ord("v"):
                gate.stop()
                if last_packet:
                    output.emit(inhibit_packet(last_packet, seq+1, "Choosing another camera"))
                    last_packet = None
                camera.close()
                detector.close()
                camera = detector = None
                picked = choose_camera(LatestCamera, args)
                if picked is None:
                    break
                choice, camera = picked
                detector = AsyncHands(model, args.confidence)
                gate, capture = StreamGate(), CalibrationCapture()
                calibration = opened = None
                expected = None if hand_mode == "auto" else hand_mode.title()
                measurement = image_points = preview_points = preview_measurement = None
                world_raw = world_filtered = source_frame = score = None
                filter_ = OneEuro(args.min_cutoff, args.beta)
                session, session_wall_ns = str(uuid.uuid4()), time.time_ns()
                frame_id = processed_id = seq = 0
                last_published_frame = None
                good_times.clear()
                reason, note = "Camera changed - show one hand", "Camera changed. Open your hand and press O."
                next_tick = next_publish = time.perf_counter()
                continue
            # Inputs reset above must also invalidate this iteration's cached validity.
            publish_ns = time.perf_counter_ns()
            age_ms = (publish_ns - source_frame.received_ns) / 1e6 if source_frame else None
            valid = measurement is not None and age_ms is not None and age_ms <= args.max_age_ms
            if not valid:
                gate.observe(False, publish_ns / 1e9)
            state = gate.state(calibration is not None, capture.active)
            if time.perf_counter() >= next_publish:
                seq += 1
                source_id = source_frame.index if source_frame else None
                new_for_packet = source_id is not None and source_id != last_published_frame
                packet = {"schema": "quadradex.hand.v1", "session_id": session, "sequence": seq,
                          "source_frame_id": source_id, "new_result": new_for_packet,
                          "clock": CLOCK_ID,
                          "source_received_monotonic_ns": source_frame.received_ns if source_frame else None,
                          "published_monotonic_ns": publish_ns, "session_started_unix_ns": session_wall_ns,
                          "state": state, "tracking_valid": bool(valid), "reason": reason,
                          "handedness": expected, "handedness_score": score if valid else None,
                          "camera": {"index": choice.index, "name": choice.name, "backend": choice.backend},
                          "source_image_size_wh": [source_frame.image.shape[1], source_frame.image.shape[0]] if source_frame else None,
                          "calibrated": calibration is not None, "calibration_progress": capture.status(now),
                          "max_age_ms": args.max_age_ms,
                          "timing": {**metrics, "result_age_ms": age_ms,
                                     "camera_read_ms": source_frame.read_ms if source_frame else None},
                          "dropped_udp": output.dropped_udp, "dropped_log_records": output.dropped_logs,
                          "landmarks_image_normalized": image_points.tolist() if valid else None,
                          "landmarks_world_model_m_raw": world_raw.tolist() if valid else None,
                          "landmarks_world_model_m_filtered": world_filtered.tolist() if valid else None,
                          "measurement": measurement if valid else None,
                          "teleop_targets": calibration.targets(measurement) if valid and gate.active and calibration else None}
                output.emit(packet)
                last_packet, last_published_frame = packet, source_id
                if new_for_packet and valid:
                    ages.append(age_ms)
                next_publish += 1. / args.output_hz
                if next_publish < time.perf_counter():
                    next_publish = time.perf_counter() + 1. / args.output_hz
            if output.log_error:
                raise RuntimeError("Log writer failed: " + output.log_error)
            if args.headless and now - last_console >= 1:
                print(f"{state}: {reason}; fresh results {metrics['valid_result_hz']:.1f} Hz", flush=True)
                last_console = now
            # Process inference results sooner than the observation publication cadence.
            next_tick += 1. / max(60., args.output_hz)
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        pass
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        if last_packet:
            output.emit(inhibit_packet(last_packet, seq+1))
        if camera:
            camera.close()
        if detector:
            detector.close()
        output.close()
        if not args.headless:
            cv2.destroyAllWindows()
        if inference_times:
            print(f"Completed valid samples: {len(inference_times)} | host receipt to inference p50/p95: "
                  f"{np.percentile(inference_times, 50):.1f}/{np.percentile(inference_times, 95):.1f} ms")
        if ages:
            print(f"Host receipt to publication p50/p95: {np.percentile(ages, 50):.1f}/{np.percentile(ages, 95):.1f} ms")
        print(f"Dropped UDP/log records: {output.dropped_udp}/{output.dropped_logs}")
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
