"""Two-camera mode: calibration from the operator's own body, and stereo fusion.

Calibration (no printed target needed)
--------------------------------------
While the operator moves both arms in view of both cameras, the 2D pose and
hand landmarks that both cameras see confidently are collected as point
correspondences. The relative camera pose is estimated from the essential
matrix (OpenCV, RANSAC) and refined by robust Levenberg-Marquardt on the
reprojection error (camera-2 rotation, translation direction and both focal
lengths, with a weak prior on a typical webcam field of view). Lens distortion
is neglected. Metric scale comes from the operator's shoulder width as
estimated by MediaPipe. The result is saved and reused while the same two
cameras are used at the same resolution.

Fusion (each frame)
-------------------
Each needed landmark is triangulated from both views when both see it
confidently and the two views agree (small reprojection error). When one view
loses it - typically the arm or fingers overlapping in that view - it is
reconstructed from the other camera's viewing ray, placed using the bone
direction from the already-triangulated parent joint (from that camera's own 3D
estimate) or the previous fused position. The hand uses triangulated landmarks
when they are geometrically plausible; otherwise the finger shape comes from
the view that sees the palm most squarely, rotated into the common frame. All
fused points are expressed in camera-1 axes (x right, y down, z away; keep
camera 1 roughly upright), the same convention as the single-camera pipeline,
so the arm/hand descriptions downstream are identical.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"cv_hand_tracking"))
from hand_geometry import features, palm_frame
from teleoperation.arm_tracking import CHAINS, IMAGE_MARGIN, ArmContinuity, LandmarkFilter, arm_geometry

DEFAULT_HFOV_DEG = 70.
HAND_ID = 100                  # correspondence ids: pose 0..32, hand 100..120
CALIB_MIN_CONFIDENCE = .75
USE_MIN_CONFIDENCE = .35
MAX_PAIR_SKEW_S = .05
PRIOR_AGE_S = .6
PARENTS = {12: 11, 11: 12, 14: 12, 13: 11, 16: 14, 15: 13}
LABELS = {"shoulder": "Shoulder", "other": "Other shoulder", "elbow": "Elbow", "wrist": "Wrist"}


# ---------------------------------------------------------------------- geometry
def default_K(size, hfov_deg=DEFAULT_HFOV_DEG):
    width, height = size
    focal = width/(2*np.tan(np.radians(hfov_deg)/2))
    return np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])


def normalize(uv, K):
    uv = np.atleast_2d(np.asarray(uv, float))
    return np.column_stack(((uv[:, 0]-K[0, 2])/K[0, 0], (uv[:, 1]-K[1, 2])/K[1, 1]))


def triangulate_normalized(x1, x2, R, t):
    """Linear (DLT) triangulation of normalized image points; camera 1 = [I|0], camera 2 = [R|t]."""
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, np.reshape(t, (3, 1))))
    A = np.stack([x1[:, 0:1]*P1[2]-P1[0], x1[:, 1:2]*P1[2]-P1[1],
                  x2[:, 0:1]*P2[2]-P2[0], x2[:, 1:2]*P2[2]-P2[1]], axis=1)
    _, _, vt = np.linalg.svd(A)
    X = vt[:, -1]
    return X[:, :3]/X[:, 3:4]


def project(K, R, t, X):
    camera = X@np.asarray(R).T+np.asarray(t)
    z = np.where(np.abs(camera[:, 2]) < 1e-9, 1e-9, camera[:, 2])
    return np.column_stack((K[0, 0]*camera[:, 0]/z+K[0, 2], K[1, 1]*camera[:, 1]/z+K[1, 2])), camera[:, 2]


def rodrigues(vector):
    return cv2.Rodrigues(np.asarray(vector, float).reshape(3, 1))[0]


def direction(theta, phi):
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])


# ---------------------------------------------------------------------- calibration result
@dataclass
class StereoCalibration:
    camera_names: list
    image_sizes: list
    K1: list
    K2: list
    R: list                  # X_cam2 = R @ X_cam1 + t  (metres)
    t: list
    rms_px: float
    inlier_fraction: float
    correspondences: int
    samples: int
    baseline_m: float
    angle_deg: float
    quality: str
    created_utc: str = ""
    schema_version: int = 1

    @property
    def k1(self):
        return np.asarray(self.K1)

    @property
    def k2(self):
        return np.asarray(self.K2)

    @property
    def r(self):
        return np.asarray(self.R)

    @property
    def tvec(self):
        return np.asarray(self.t)

    def camera2_center(self):
        return -self.r.T@self.tvec

    def matches(self, names, sizes):
        return list(self.camera_names) == list(names) and [list(s) for s in self.image_sizes] == [list(s) for s in sizes]

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != 1:
            raise ValueError("Incompatible stereo calibration file")
        calibration = cls(**data)
        if np.asarray(calibration.R).shape != (3, 3) or not np.isfinite(np.asarray(calibration.t)).all():
            raise ValueError("Invalid stereo calibration file")
        return calibration

    def triangulate(self, uv1, uv2):
        """Pixel points (n,2) in each image -> 3D points in camera-1 axes, and per-view reprojection errors."""
        x1, x2 = normalize(uv1, self.k1), normalize(uv2, self.k2)
        X = triangulate_normalized(x1, x2, self.r, self.tvec)
        p1, z1 = project(self.k1, np.eye(3), np.zeros(3), X)
        p2, z2 = project(self.k2, self.r, self.tvec, X)
        e1 = np.linalg.norm(p1-np.atleast_2d(uv1), axis=1)
        e2 = np.linalg.norm(p2-np.atleast_2d(uv2), axis=1)
        behind = (z1 <= 0) | (z2 <= 0)
        e1[behind] = e2[behind] = np.inf
        return X, e1, e2

    def ray(self, view, uv):
        """(origin, unit direction) of the viewing ray through pixel uv of camera `view` (1 or 2), camera-1 axes."""
        K = self.k1 if view == 1 else self.k2
        x = normalize(uv, K)[0]
        d = np.array([x[0], x[1], 1.])
        if view == 1:
            return np.zeros(3), d/np.linalg.norm(d)
        d = self.r.T@d
        return self.camera2_center(), d/np.linalg.norm(d)

    def to_camera1(self, view):
        """Rotation taking camera-`view` axes to camera-1 axes."""
        return np.eye(3) if view == 1 else self.r.T


def _refine(x_px1, x_px2, K1, K2, R, t, iterations=40):
    """Robust LM over [log f1, log f2, rotation vector, translation direction]."""
    f01, f02 = K1[0, 0], K2[0, 0]
    rvec = cv2.Rodrigues(R)[0].ravel()
    t = t/np.linalg.norm(t)
    params = np.r_[np.log(f01), np.log(f02), rvec, np.arccos(np.clip(t[2], -1, 1)), np.arctan2(t[1], t[0])]
    n = len(x_px1)
    prior = np.sqrt(n)/.3

    def unpack(p):
        k1, k2 = K1.copy(), K2.copy()
        k1[0, 0] = k1[1, 1] = np.exp(p[0])
        k2[0, 0] = k2[1, 1] = np.exp(p[1])
        return k1, k2, rodrigues(p[2:5]), direction(p[5], p[6])

    def residual(p, weights=None):
        k1, k2, r, tt = unpack(p)
        X = triangulate_normalized(normalize(x_px1, k1), normalize(x_px2, k2), r, tt)
        p1, z1 = project(k1, np.eye(3), np.zeros(3), X)
        p2, z2 = project(k2, r, tt, X)
        err = np.hstack((p1-x_px1, p2-x_px2))
        err[(z1 <= 0) | (z2 <= 0)] = 50.
        if weights is not None:
            err = err*weights[:, None]
        return np.r_[err.ravel(), prior*(p[0]-np.log(f01)), prior*(p[1]-np.log(f02))], err

    damping = 1e-3
    for _ in range(iterations):
        _, raw = residual(params)
        point_error = np.linalg.norm(raw.reshape(n, 2, 2), axis=2).max(axis=1)
        weights = np.sqrt(np.minimum(1., 4./np.maximum(point_error, 1e-9)))   # Huber-like, 4 px
        r0, _ = residual(params, weights)
        jac = np.empty((len(r0), len(params)))
        for i in range(len(params)):
            step = np.zeros_like(params)
            step[i] = 1e-6*max(1., abs(params[i]))
            jac[:, i] = (residual(params+step, weights)[0]-r0)/step[i]
        while True:
            h = jac.T@jac
            delta = np.linalg.solve(h+damping*np.diag(np.diag(h)+1e-9), -jac.T@r0)
            candidate = params+delta
            if np.sum(residual(candidate, weights)[0]**2) < np.sum(r0**2):
                params, damping = candidate, max(damping/3, 1e-7)
                break
            damping *= 4
            if damping > 1e6:
                return unpack(params)
        if np.linalg.norm(delta) < 1e-8:
            break
    return unpack(params)


def calibrate(samples, names, sizes, hfov_deg=DEFAULT_HFOV_DEG):
    """samples: [{'uv1': {id: (u, v)}, 'uv2': {...}, 'shoulder_width_m': float|None}] -> StereoCalibration."""
    keys, uv1, uv2, which = [], [], [], []
    for index, sample in enumerate(samples):
        for key in sorted(set(sample['uv1']) & set(sample['uv2'])):
            uv1.append(sample['uv1'][key])
            uv2.append(sample['uv2'][key])
            keys.append(key)
            which.append(index)
    uv1, uv2 = np.asarray(uv1, float), np.asarray(uv2, float)
    if len(uv1) < 60:
        raise ValueError("Too few points seen by both cameras; make sure both cameras see your upper body and arms")
    K1, K2 = default_K(sizes[0], hfov_deg), default_K(sizes[1], hfov_deg)
    x1, x2 = normalize(uv1, K1), normalize(uv2, K2)
    E, mask = cv2.findEssentialMat(x1, x2, np.eye(3), method=cv2.RANSAC, prob=.999, threshold=5./K1[0, 0])
    if E is None or E.shape != (3, 3):
        raise ValueError("Could not relate the two camera views; move your arms more, in depth as well as sideways")
    _, R, t, pose_mask = cv2.recoverPose(E, x1, x2, np.eye(3), mask=mask.copy())
    inliers = pose_mask.ravel() > 0
    if inliers.sum() < 40:
        raise ValueError("The two views disagree; check that both cameras see the same person and neither image is mirrored")
    chosen = np.flatnonzero(inliers)
    if len(chosen) > 1500:
        chosen = np.random.default_rng(0).choice(chosen, 1500, replace=False)
    K1, K2, R, t = _refine(uv1[chosen], uv2[chosen], K1, K2, R, t.ravel())
    probe = StereoCalibration(list(names), [list(s) for s in sizes], K1.tolist(), K2.tolist(), R.tolist(), t.tolist(),
                              0., 0., 0, 0, 0., 0., "")
    _, e1, e2 = probe.triangulate(uv1, uv2)
    error = np.maximum(e1, e2)
    good = error < max(10., 3*np.median(error))
    rms = float(np.sqrt(np.mean(np.r_[e1[good], e2[good]]**2)))
    # Metric scale from the operator's shoulder width.
    ratios = []
    for index, sample in enumerate(samples):
        width = sample.get('shoulder_width_m')
        if width and 11 in sample['uv1'] and 12 in sample['uv1'] and 11 in sample['uv2'] and 12 in sample['uv2']:
            X, a, b = probe.triangulate(np.array([sample['uv1'][11], sample['uv1'][12]]), np.array([sample['uv2'][11], sample['uv2'][12]]))
            if np.isfinite(a).all() and np.isfinite(b).all():
                ratios.append(width/max(np.linalg.norm(X[0]-X[1]), 1e-9))
    if len(ratios) < 5:
        raise ValueError("Both shoulders must be visible to both cameras during the setup")
    scale = float(np.median(ratios))
    t = t*scale
    angle = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    quality = "good" if rms < 6 and angle >= 20 else "usable" if rms < 12 and angle >= 10 else "poor"
    return StereoCalibration(list(names), [list(s) for s in sizes], K1.tolist(), K2.tolist(), R.tolist(), t.tolist(),
                             rms, float(good.mean()), int(len(uv1)), len(samples), float(np.linalg.norm(t)), angle, quality,
                             datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------- pairing and collection
def pair_observations(history1, history2, max_skew_s=MAX_PAIR_SKEW_S):
    """Most recent pair of observations from the two cameras captured close together in time.

    Returns (obs1, obs2, skew_s); either observation may be None when no pair is close enough
    (then the more recent one is returned alone)."""
    if not history1 or not history2:
        return (history1[-1] if history1 else None), (history2[-1] if history2 else None), None
    best = None
    for a in history1[-4:]:
        for b in history2[-4:]:
            skew = abs(a.frame.received_ns-b.frame.received_ns)/1e9
            older = min(a.frame.received_ns, b.frame.received_ns)
            key = (skew > max_skew_s, -older if skew <= max_skew_s else skew, skew)
            if best is None or key < best[0]:
                best = (key, a, b, skew)
    _, a, b, skew = best
    if skew <= max_skew_s:
        return a, b, skew
    if history1[-1].frame.received_ns >= history2[-1].frame.received_ns:
        return history1[-1], None, skew
    return None, history2[-1], skew


def view_points(observation, min_confidence=CALIB_MIN_CONFIDENCE, include_hand=True):
    """{id: (u, v) pixels} of confidently seen landmarks in one view."""
    points = {}
    if observation is None or observation.pose_image is None or observation.pose_confidence is None:
        return points
    width, height = observation.image_size
    image = observation.pose_image
    for index in range(33):
        x, y = image[index, :2]
        if observation.pose_confidence[index] >= min_confidence and 0 <= x <= 1 and 0 <= y <= 1:
            points[index] = (x*width, y*height)
    if include_hand and observation.hand_image is not None and observation.hand_world is not None:
        for k in range(21):
            x, y = observation.hand_image[k, :2]
            if 0 <= x <= 1 and 0 <= y <= 1:
                points[HAND_ID+k] = (x*width, y*height)
    return points


class StereoCollector:
    """Gathers correspondences for calibration while the operator moves."""

    def __init__(self, target_samples=140, min_interval_s=.08, max_skew_s=.03):
        self.target, self.min_interval, self.max_skew = target_samples, min_interval_s, max_skew_s
        self.samples = []
        self.last_time = None
        self.last_wrists = None
        self.message = "Stand where both cameras see your head, shoulders and arms. Move both arms slowly in all directions."

    @property
    def progress(self):
        return min(1., len(self.samples)/self.target)

    @property
    def complete(self):
        return len(self.samples) >= self.target

    def update(self, obs1, obs2, skew_s, now):
        if obs1 is None or obs2 is None or skew_s is None or skew_s > self.max_skew:
            self.message = "Waiting for both cameras to see you at the same moment..."
            return False
        if self.last_time is not None and now-self.last_time < self.min_interval:
            return False
        uv1, uv2 = view_points(obs1), view_points(obs2)
        common = set(uv1) & set(uv2)
        if len([k for k in common if k < HAND_ID]) < 6 or not {11, 12} <= common:
            self.message = "Both cameras must see your shoulders and arms clearly (step back or turn the cameras)"
            return False
        wrists = np.array([uv1.get(15, (0, 0)), uv1.get(16, (0, 0))])
        if self.last_wrists is not None and np.max(np.abs(wrists-self.last_wrists)) < 4 and len(self.samples) > 20:
            self.message = "Keep moving your arms (forward/back as well as up/down and sideways)"
            return False
        widths = [np.linalg.norm(o.pose_world[11]-o.pose_world[12]) for o in (obs1, obs2) if o.pose_world is not None]
        self.samples.append({'uv1': {k: uv1[k] for k in common}, 'uv2': {k: uv2[k] for k in common},
                             'shoulder_width_m': float(np.mean(widths)) if widths else None})
        self.last_time, self.last_wrists = now, wrists
        self.message = f"Collecting: {len(self.samples)}/{self.target} - keep moving both arms slowly, in depth too"
        return True


# ---------------------------------------------------------------------- fusion
@dataclass
class FusionResult:
    arm: dict | None
    hand: dict | None
    parts: list
    arm_reason: str
    hand_reason: str
    info: dict = field(default_factory=dict)


class StereoFusion:
    def __init__(self, calibration, side="right", min_visibility=USE_MIN_CONFIDENCE):
        self.calibration = calibration
        self.side = side
        self.min_confidence = min_visibility
        self.max_error = max(12., 3*calibration.rms_px)
        self.arm_filter = LandmarkFilter(1.2, 5.0, reset_gap_s=.5)
        self.hand_filter = LandmarkFilter(2.5, 10.0, reset_gap_s=.3)
        self.continuity = ArmContinuity()
        self.previous = {}
        self.hand_source = None
        self.recent_errors = []

    def _view(self, observation, index):
        if observation is None or observation.pose_image is None or observation.pose_confidence is None:
            return None, 0.
        x, y = observation.pose_image[index, :2]
        inside = -IMAGE_MARGIN <= x <= 1+IMAGE_MARGIN and -IMAGE_MARGIN <= y <= 1+IMAGE_MARGIN
        width, height = observation.image_size
        return np.array([x*width, y*height]), float(observation.pose_confidence[index]) if inside else 0.

    def _single(self, view, observation, uv, index, points, now):
        """Point on the viewing ray, placed by the parent joint + this view's own bone vector, or the previous point."""
        origin, ray = self.calibration.ray(view, uv)
        parent = PARENTS.get(index)
        guess = None
        if parent in points and observation.pose_world is not None:
            bone = self.calibration.to_camera1(view)@(observation.pose_world[index]-observation.pose_world[parent])
            guess = points[parent]+bone
        elif index in self.previous and now-self.previous[index][1] < PRIOR_AGE_S:
            guess = self.previous[index][0]
        if guess is None:
            return None
        return origin+ray*max(np.dot(guess-origin, ray), .05)

    def fuse(self, obs1, obs2, now):
        shoulder, elbow, wrist = CHAINS[self.side]
        order = [(shoulder, "shoulder"), (11 if self.side == "right" else 12, "other"), (elbow, "elbow"), (wrist, "wrist")]
        points, sources, scores, errors = {}, {}, {}, []
        for index, name in order:
            uv1, c1 = self._view(obs1, index)
            uv2, c2 = self._view(obs2, index)
            use1, use2 = c1 >= self.min_confidence, c2 >= self.min_confidence
            point = None
            if use1 and use2:
                X, e1, e2 = self.calibration.triangulate(uv1[None], uv2[None])
                if max(e1[0], e2[0]) <= self.max_error:
                    point, sources[name] = X[0], "both cameras"
                    errors.append(max(e1[0], e2[0]))
                else:
                    use1, use2 = c1 >= c2, c2 > c1   # the views disagree: trust the more confident one
            if point is None and (use1 or use2):
                view, obs, uv = (1, obs1, uv1) if use1 else (2, obs2, uv2)
                point = self._single(view, obs, uv, index, points, now)
                if point is not None:
                    sources[name] = f"camera {view} only"
            if point is not None:
                points[index] = point
                self.previous[index] = (point, now)
            scores[name] = max(c1, c2)
        parts = [(LABELS[name], index in points, scores[name], sources.get(name, "hidden in both cameras"))
                 for index, name in order]
        self.recent_errors = (self.recent_errors+errors)[-120:]
        arm = hand = None
        arm_reason = hand_reason = ""
        source_s = now
        if all(index in points for index, _ in order):
            world = np.zeros((33, 3))
            ids = [11, 12, elbow, wrist]
            world[ids] = self.arm_filter.update(np.array([points[i] for i in ids]), source_s)
            try:
                arm = arm_geometry(world, self.side, min(scores.values()))
                self.continuity.check(arm, source_s)
                arm["side"] = self.side.title()
                arm["stereo_sources"] = {name: sources.get(name) for _, name in order}
            except ValueError as error:
                arm, arm_reason = None, str(error)
        else:
            missing = [LABELS[name].lower() for index, name in order if index not in points]
            arm_reason = f"{self.side.title()} arm: {', '.join(missing)} not visible to either camera"
        hand, hand_reason, hand_info = self._fuse_hand(obs1, obs2, source_s)
        info = {"pose_sources": sources, "hand": hand_info,
                "median_reprojection_px": float(np.median(self.recent_errors)) if self.recent_errors else None}
        return FusionResult(arm, hand, parts, arm_reason, hand_reason, info)

    def _fuse_hand(self, obs1, obs2, source_s):
        views = [(v, o) for v, o in ((1, obs1), (2, obs2)) if o is not None and o.hand_image is not None and o.hand_world is not None]
        if not views:
            self.hand_filter.reset()
            reasons = [o.hand_reason for o in (obs1, obs2) if o is not None and o.hand_reason]
            return None, (reasons[0] if reasons else "Hand not visible to either camera"), {"source": None}
        candidate, source, info = None, None, {}
        if len(views) == 2:
            (_, a), (_, b) = views
            wa, ha = a.image_size
            wb, hb = b.image_size
            X, e1, e2 = self.calibration.triangulate(a.hand_image[:, :2]*[wa, ha], b.hand_image[:, :2]*[wb, hb])
            error = float(np.median(np.maximum(e1, e2)))
            info["triangulation_px"] = error
            if np.isfinite(error) and error <= self.max_error:
                try:
                    _, width = palm_frame(X)
                    lengths = np.linalg.norm(np.diff(X[[0, 5, 6, 7, 8]], axis=0), axis=1)/width
                    if .04 <= width <= .14 and np.all((lengths > .1) & (lengths < 1.6)):
                        candidate, source = X, "triangulated"
                except ValueError:
                    pass
        if candidate is None:
            def frontal(item):
                view, observation = item
                try:
                    normal = palm_frame(observation.hand_world)[0][:, 2]
                except ValueError:
                    return -1.
                return abs(normal[2])
            scored = sorted(views, key=frontal, reverse=True)
            best = scored[0]
            if self.hand_source in ("camera 1", "camera 2") and len(scored) == 2:
                current = next((item for item in scored if f"camera {item[0]}" == self.hand_source), best)
                if frontal(best)-frontal(current) < .15:
                    best = current   # hysteresis against flicker between views
            view, observation = best
            candidate = observation.hand_world@self.calibration.to_camera1(view).T
            source = f"camera {view}"
        self.hand_source = source
        info["source"] = source
        try:
            # Wrist-centred: only the hand's shape and orientation matter, and switching
            # between triangulated and single-view candidates must not jump.
            return features(self.hand_filter.update(candidate-candidate[0], source_s)), "", info
        except ValueError as error:
            self.hand_filter.reset()
            return None, str(error), info
