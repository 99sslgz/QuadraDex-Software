"""Camera-independent geometry and calibration for the QuadraDex S2 prototype.

All 3D inputs are MediaPipe model estimates in metres, not measured camera depth.
No motor indices, robot joint limits, or controller conventions are assumed here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


FINGERS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8),
           "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
           "pinky": (17, 18, 19, 20)}
CONTROLLED = ("thumb", "index", "middle", "ring")
CONNECTIONS = [(0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (17, 0)] + [
    (a, b) for chain in FINGERS.values() for a, b in zip(chain, chain[1:])]


def unit(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-8:
        raise ValueError("Degenerate hand geometry")
    return vector / length


def bend(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Unsigned bend: a straight joint is 0 degrees, a right angle is 90."""
    cosine = np.clip(np.dot(unit(b - a), unit(c - b)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def palm_frame(world: np.ndarray) -> tuple[np.ndarray, float]:
    """Return columns [radial, distal, radial x distal] in model camera axes."""
    radial = unit(world[5] - world[17])
    distal = world[9] - world[0]
    distal = unit(distal - np.dot(distal, radial) * radial)
    normal = unit(np.cross(radial, distal))
    return np.column_stack((radial, distal, normal)), float(np.linalg.norm(world[5] - world[17]))


def validate_geometry(image: np.ndarray, world: np.ndarray) -> str | None:
    """Reject gross failure only. Passing does not prove landmark accuracy."""
    if image.shape != (21, 3) or world.shape != (21, 3):
        return "Incomplete landmarks"
    if not np.isfinite(image).all() or not np.isfinite(world).all():
        return "Non-finite landmarks"
    if np.any(image[:, :2] < 0) or np.any(image[:, :2] > 1):
        return "Hand is partly outside the camera image"
    try:
        _, width = palm_frame(world)
        if not 0.025 <= width <= 0.16:
            return "Implausible palm width"
        for a, b in CONNECTIONS:
            ratio = np.linalg.norm(world[b] - world[a]) / width
            if not 0.035 <= ratio <= 2.8:
                return "Implausible bone length"
    except ValueError as error:
        return str(error)
    return None


class OneEuro:
    """Adaptive low-pass filter on 3D landmarks; reset after invalid input."""

    def __init__(self, min_cutoff: float = 2.0, beta: float = 12.0):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.time = None
        self.raw = self.filtered = self.derivative = None

    @staticmethod
    def alpha(cutoff, dt):
        return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))

    def update(self, value: np.ndarray, timestamp_s: float) -> np.ndarray:
        if self.time is None:
            self.time = timestamp_s
            self.raw, self.filtered = value.copy(), value.copy()
            self.derivative = np.zeros_like(value)
            return value.copy()
        dt = timestamp_s - self.time
        if dt <= 0:
            raise ValueError("Filter timestamps must increase")
        derivative = (value - self.raw) / dt
        a = self.alpha(1.0, dt)
        self.derivative = a * derivative + (1 - a) * self.derivative
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.derivative, axis=1, keepdims=True)
        a = self.alpha(cutoff, dt)
        self.filtered = a * value + (1 - a) * self.filtered
        self.time, self.raw = timestamp_s, value.copy()
        return self.filtered.copy()


def features(world: np.ndarray) -> dict:
    basis, width = palm_frame(world)
    local = (world - world[0]) @ basis
    joints, curls, tips, directions = {}, {}, {}, {}
    for name, (base, proximal, distal, tip) in FINGERS.items():
        if name == "thumb":
            angles = {"cmc_bend_proxy": bend(world[0], world[base], world[proximal]),
                      "mcp": bend(world[base], world[proximal], world[distal]),
                      "ip": bend(world[proximal], world[distal], world[tip])}
            curls[name] = (angles["mcp"] + angles["ip"]) / 180.0
        else:
            # Resolve MCP out-of-palm bend separately from in-palm splay.
            metacarpal = local[base].copy()
            metacarpal[2] = 0
            longitudinal = unit(metacarpal)
            lateral = np.cross(longitudinal, np.array([0., 0., 1.]))
            segment = unit(local[proximal] - local[base])
            forward = float(np.dot(segment, longitudinal))
            angles = {"mcp": float(np.degrees(np.arctan2(abs(segment[2]), forward))),
                      "pip": bend(world[base], world[proximal], world[distal]),
                      "dip": bend(world[proximal], world[distal], world[tip]),
                      "splay": float(np.degrees(np.arctan2(np.dot(segment, lateral), forward)))}
            curls[name] = (angles["mcp"] + angles["pip"] + angles["dip"]) / 270.0
        joints[name] = angles
        tips[name] = local[tip].tolist()
        directions[name] = unit(local[tip] - local[distal]).tolist()
    return {"joints_deg": joints,
            "curl_raw": {k: float(np.clip(v, 0, 1)) for k, v in curls.items()},
            "palm_width_model_m": width,
            "palm_basis_columns_in_model_camera": basis.tolist(),
            "landmarks_palm_model_m": local.tolist(),
            "fingertips_palm_model_m": tips,
            "distal_directions_palm": directions,
            "pinch_distance_palm_widths": {
                name: float(np.linalg.norm(local[4] - local[FINGERS[name][-1]]) / width)
                for name in ("index", "middle", "ring", "pinky")}}


@dataclass
class Calibration:
    handedness: str
    open_curl: dict[str, float]
    closed_curl: dict[str, float]
    palm_width_model_m: float
    created_utc: str
    schema_version: int = 1

    def validate(self):
        if self.schema_version != 1 or self.handedness not in ("Left", "Right"):
            raise ValueError("Incompatible calibration profile")
        if not np.isfinite(self.palm_width_model_m) or not 0.025 <= self.palm_width_model_m <= 0.16:
            raise ValueError("Invalid calibrated palm width")
        for finger in FINGERS:
            a, b = self.open_curl[finger], self.closed_curl[finger]
            if not np.isfinite([a, b]).all() or not 0 <= a <= 1 or not 0 <= b <= 1:
                raise ValueError("Invalid calibration curl")
            if finger in CONTROLLED and b - a < 0.10:
                raise ValueError(f"Insufficient open/closed motion for {finger}; bend it further")

    def save(self, path: Path):
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path, handedness: str | None):
        profile = cls(**json.loads(path.read_text(encoding="utf-8")))
        profile.validate()
        if handedness is not None and profile.handedness != handedness:
            raise ValueError("Calibration belongs to the other hand")
        return profile

    def targets(self, measurement: dict) -> dict:
        self.validate()
        curl = {}
        for name in FINGERS:
            span = self.closed_curl[name] - self.open_curl[name]
            curl[name] = float(np.clip((measurement["curl_raw"][name] - self.open_curl[name]) / span, 0, 1)) if span >= .10 else None
        width = self.palm_width_model_m
        tips = {name: (np.asarray(measurement["fingertips_palm_model_m"][name]) / width).tolist()
                for name in CONTROLLED}
        return {"frame": "wrist_origin_radial_distal_cross",
                "units": "calibrated_palm_widths", "finger_order": list(CONTROLLED),
                "fingertips": tips,
                "thumb_to_fingertip_vectors": {
                    name: (np.asarray(tips[name]) - tips["thumb"]).tolist()
                    for name in CONTROLLED if name != "thumb"},
                "distal_directions": {name: measurement["distal_directions_palm"][name] for name in CONTROLLED},
                "curl_01": {name: curl[name] for name in CONTROLLED}}


def calibration_pose(samples: list[dict]) -> dict:
    if len(samples) < 8:
        raise ValueError("Too few valid samples; improve visibility and try again")
    curls = np.array([[sample["curl_raw"][name] for name in FINGERS] for sample in samples])
    widths = np.array([sample["palm_width_model_m"] for sample in samples])
    controlled_columns = [list(FINGERS).index(finger) for finger in CONTROLLED]
    if np.max(np.std(curls[:, controlled_columns], axis=0)) > 0.07 or np.std(widths) / np.mean(widths) > 0.10:
        raise ValueError("Hand moved during calibration; hold the pose steady")
    return {"curl": dict(zip(FINGERS, np.median(curls, axis=0).tolist())),
            "width": float(np.median(widths))}


def make_calibration(handedness: str, opened: dict, closed: dict) -> Calibration:
    if abs(opened["width"] - closed["width"]) / opened["width"] > 0.25:
        raise ValueError("Palm estimates changed too much between poses")
    profile = Calibration(handedness, opened["curl"], closed["curl"], opened["width"],
                          datetime.now(timezone.utc).isoformat())
    profile.validate()
    return profile


class StreamGate:
    """Local target-stream permission, separate from downstream robot permission."""

    def __init__(self, recovery_s: float = 0.5):
        self.active = False
        self.fault = False
        self.valid_since = None
        self.recovery_s = recovery_s

    def observe(self, valid: bool, now: float):
        if not valid:
            self.fault = self.fault or self.active
            self.active = False
            self.valid_since = None
        elif self.valid_since is None:
            self.valid_since = now

    def start(self, calibrated: bool, valid: bool, now: float) -> bool:
        if not calibrated or not valid or self.valid_since is None:
            return False
        if now - self.valid_since < self.recovery_s:
            return False
        self.active, self.fault = True, False
        return True

    def stop(self):
        self.active = False

    def state(self, calibrated: bool, calibrating: bool = False) -> str:
        if calibrating:
            return "CALIBRATING"
        if self.fault:
            return "TRACKING_LOST"
        if not calibrated:
            return "CALIBRATION_REQUIRED"
        return "STREAMING" if self.active else "READY"
