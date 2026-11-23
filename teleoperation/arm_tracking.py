"""Shoulder/elbow/wrist estimates and a torso-relative operator calibration.

Only the two shoulders and the selected elbow/wrist are required. Hips are often
hidden by a desk or outside a webcam's field of view, so the body frame uses the
shoulder line for heading and the camera's vertical for "up" (keep the camera
roughly upright). MediaPipe pose landmarks are anatomical (11 = the person's
left shoulder) for an unmirrored camera image.
"""
from dataclasses import dataclass, asdict
import numpy as np

SHOULDERS = (11, 12)
CHAINS = {"right": (12, 14, 16), "left": (11, 13, 15)}
CAMERA_UP = np.array([0., -1., 0.])  # MediaPipe world axes: x right, y down, z away from camera
IMAGE_MARGIN = .03


def unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-7:
        raise ValueError("Degenerate arm or torso geometry")
    return vector / length


def pose_arrays(result):
    if not result.pose_landmarks or not result.pose_world_landmarks:
        raise ValueError("No person detected: face the camera with your head, shoulders and arm in view")
    image = np.array([[p.x, p.y, p.z] for p in result.pose_landmarks[0]])
    world = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]])
    confidence = np.array([min(p.visibility if p.visibility is not None else 0., p.presence if p.presence is not None else 0.)
                           for p in result.pose_landmarks[0]])
    return image, world, confidence


def part_status(image, confidence, side, min_visibility=.5):
    """Per-landmark readiness for the operator display: (label, ok, score, hint)."""
    shoulder, elbow, wrist = CHAINS[side]
    other = 11 if side == "right" else 12
    out = []
    for label, index in (("Shoulder", shoulder), ("Other shoulder", other), ("Elbow", elbow), ("Wrist", wrist)):
        if image is None:
            out.append((label, False, 0., "no person detected"))
            continue
        score = float(confidence[index])
        x, y = (float(v) for v in image[index, :2])  # plain floats -> plain bools (JSON-safe)
        inside = -IMAGE_MARGIN <= x <= 1+IMAGE_MARGIN and -IMAGE_MARGIN <= y <= 1+IMAGE_MARGIN
        ok = bool(inside and score >= min_visibility)
        hint = "" if ok else ("outside the image" if not inside else "not clearly visible")
        out.append((label, ok, score, hint))
    return out


def torso_basis(world):
    """Columns [forward, left, up] in MediaPipe world axes, from the shoulder line and camera up."""
    across = world[11] - world[12]  # person's right -> left shoulder
    across = across - np.dot(across, CAMERA_UP) * CAMERA_UP
    left = unit(across)
    up = unit(CAMERA_UP - np.dot(CAMERA_UP, left) * left)
    forward = unit(np.cross(left, up))
    return np.column_stack((forward, left, up))


def arm_features(image, world, confidence, side="right", min_visibility=.5):
    if image.shape != (33, 3) or world.shape != (33, 3) or confidence.shape != (33,):
        raise ValueError("Incomplete body landmarks")
    if not np.isfinite(image).all() or not np.isfinite(world).all() or not np.isfinite(confidence).all():
        raise ValueError("Non-finite body landmarks")
    shoulder, elbow, wrist = CHAINS[side]
    for label, ok, score, hint in part_status(image, confidence, side, min_visibility):
        if not ok:
            raise ValueError(f"{side.title()} arm: {label.lower()} {hint} ({score:.2f})")
    return arm_geometry(world, side, float(np.min(confidence[[11, 12, elbow, wrist]])))


def arm_geometry(world, side="right", minimum_visibility=1.):
    """Body-frame arm description from 3D landmarks in camera-aligned axes (x right, y down, z away).

    Used for single-camera MediaPipe world landmarks and for two-camera triangulated points.
    """
    shoulder, elbow, wrist = CHAINS[side]
    if not np.isfinite(world[[11, 12, elbow, wrist]]).all():
        raise ValueError("Non-finite body landmarks")
    upper = float(np.linalg.norm(world[elbow]-world[shoulder]))
    forearm = float(np.linalg.norm(world[wrist]-world[elbow]))
    width = float(np.linalg.norm(world[11]-world[12]))
    if not .10 < upper < .65 or not .08 < forearm < .60 or not .12 < width < .70:
        raise ValueError("Implausible arm proportions; step back so the whole arm is clearly visible")
    basis = torso_basis(world)
    origin = (world[11]+world[12])/2
    local = (world[[shoulder, elbow, wrist]]-origin)@basis
    flexion = np.degrees(np.arccos(np.clip(np.dot(unit(world[elbow]-world[shoulder]), unit(world[wrist]-world[elbow])), -1, 1)))
    return {"side": side.title(), "frame": "mid_shoulder_origin_forward_left_up",
            "shoulder_model_m": world[shoulder].tolist(), "elbow_model_m": world[elbow].tolist(), "wrist_model_m": world[wrist].tolist(),
            "points_body_m": local.tolist(), "basis_columns_in_model_camera": basis.tolist(),
            "upper_arm_model_m": upper, "forearm_model_m": forearm, "shoulder_width_model_m": width,
            "elbow_flexion_deg": float(flexion), "minimum_visibility": float(minimum_visibility),
            "accuracy_status": "not_independently_validated"}


class LandmarkFilter:
    """One-Euro filter for selected 3D landmarks that survives brief dropouts."""

    def __init__(self, min_cutoff=1.2, beta=6.0, d_cutoff=1.0, reset_gap_s=.4):
        self.min_cutoff, self.beta, self.d_cutoff, self.reset_gap_s = min_cutoff, beta, d_cutoff, reset_gap_s
        self.time = self.value = self.derivative = None

    @staticmethod
    def alpha(cutoff, dt):
        return 1.0/(1.0+1.0/(2*np.pi*cutoff*dt))

    def reset(self):
        self.time = self.value = self.derivative = None

    def update(self, value, timestamp_s):
        value = np.asarray(value, float)
        if self.time is None or timestamp_s-self.time > self.reset_gap_s or value.shape != self.value.shape:
            self.time, self.value, self.derivative = timestamp_s, value.copy(), np.zeros_like(value)
            return value.copy()
        dt = timestamp_s-self.time
        if dt <= 0:
            return self.value.copy()
        derivative = (value-self.value)/dt
        a = self.alpha(self.d_cutoff, dt)
        self.derivative = a*derivative+(1-a)*self.derivative
        cutoff = self.min_cutoff+self.beta*np.linalg.norm(self.derivative, axis=-1, keepdims=True)
        a = self.alpha(cutoff, dt)
        self.value = a*value+(1-a)*self.value
        self.time = timestamp_s
        return self.value.copy()


@dataclass
class ArmAnchor:
    side: str
    wrist_body_m: list
    elbow_body_m: list
    shoulder_body_m: list
    palm_in_body: list
    upper_arm_model_m: float
    forearm_model_m: float

    @classmethod
    def capture(cls, arms, hands, max_spread_m=.045):
        if len(arms) < 8 or len(arms) != len(hands):
            raise ValueError("Need at least eight synchronized arm/hand calibration samples")
        wrists = np.array([a["points_body_m"][2] for a in arms])
        spread = np.percentile(np.linalg.norm(wrists-np.median(wrists, axis=0), axis=1), 80)
        if spread > max_spread_m:
            raise ValueError("Arm moved during the OPEN calibration; hold it still and retry O")
        palms = []
        for arm, hand in zip(arms, hands):
            palms.append(np.asarray(arm["basis_columns_in_model_camera"]).T@np.asarray(hand["palm_basis_columns_in_model_camera"]))
        palm = average_rotation(palms)
        median = lambda key, index: np.median([a[key][index] for a in arms], axis=0).tolist()
        return cls(arms[-1]["side"], np.median(wrists, axis=0).tolist(), median("points_body_m", 1), median("points_body_m", 0),
                   palm.tolist(), float(np.median([a["upper_arm_model_m"] for a in arms])),
                   float(np.median([a["forearm_model_m"] for a in arms])))

    @property
    def arm_length_m(self):
        return self.upper_arm_model_m+self.forearm_model_m

    def validate_measurement(self, arm):
        if arm["side"] != self.side:
            raise ValueError("The calibrated arm side changed")
        total = arm["upper_arm_model_m"]+arm["forearm_model_m"]
        if abs(total/self.arm_length_m-1) > .45:
            raise ValueError("Arm length estimate changed a lot; restore a clear view or recalibrate")

    def to_dict(self):
        return asdict(self)


def average_rotation(rotations):
    """Chordal L2 mean of rotation matrices, projected back onto SO(3)."""
    u, _, vt = np.linalg.svd(np.mean(np.asarray(rotations), axis=0))
    rotation = u@vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u@vt
    return rotation


class ArmContinuity:
    """Reject physically implausible single-frame jumps of the wrist estimate."""

    def __init__(self, max_speed_m_s=4.0):
        self.max_speed = max_speed_m_s
        self.time = self.wrist = self.pending = None

    def _fast(self, wrist, source_s, reference):
        time, previous = reference
        return 0 < source_s-time < .25 and np.linalg.norm(wrist-previous)/(source_s-time) > self.max_speed

    def check(self, arm, source_s):
        wrist = np.asarray(arm["points_body_m"][2])
        if self.time is not None and self._fast(wrist, source_s, (self.time, self.wrist)):
            # A one-frame glitch is dropped. If the next estimate agrees with the
            # new location, the estimate really moved and is accepted.
            if self.pending is not None and 0 < source_s-self.pending[0] < .25 and not self._fast(wrist, source_s, self.pending):
                self.time, self.wrist, self.pending = source_s, wrist, None
                return
            self.pending = (source_s, wrist)
            raise ValueError("Sudden arm estimate jump ignored")
        self.time, self.wrist, self.pending = source_s, wrist, None
