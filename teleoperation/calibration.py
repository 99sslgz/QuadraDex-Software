"""Guided arm + hand calibration and the teleoperation stream gate.

OPEN (O): the start pose with an open hand, both held still. It sets the arm
anchor, the palm orientation reference and the open-hand finger geometry; the
robot computes a reference pose that matches it.
CLOSED (C): fist. Only the hand is required, so arm visibility cannot block it.

Recommended start pose (it gives the camera a clear view of every finger and
an elbow angle that the robot can match): upper arm relaxed down at your side,
elbow bent so the forearm points up and forward, open hand with the palm
facing the camera and the fingers pointing up (like raising your hand).
"""
import numpy as np

from hand_geometry import CONTROLLED, calibration_pose, features
from teleoperation.arm_tracking import ArmAnchor


def median_hand(hands):
    """Open-hand reference from the median palm-frame landmarks of all samples."""
    local = np.median([np.asarray(h["landmarks_palm_model_m"]) for h in hands], axis=0)
    return features(local)


class PairedCalibration:
    def __init__(self, duration_s=1.0, prepare_s=1.5, timeout_s=30., freshness_s=.2):
        self.duration_s, self.prepare_s, self.timeout_s, self.freshness_s = duration_s, prepare_s, timeout_s, freshness_s
        self.stage = None
        self.started = 0.
        self.arms, self.hands = [], []
        self.valid_s = 0.
        self.last_source_s = None
        self.message = "Step 1 of 2: neutral arm, open hand - press O"
        self.phase = "idle"

    @property
    def active(self):
        return self.stage is not None

    @property
    def samples(self):
        return self.hands

    def start(self, stage, now):
        self.stage, self.started = stage, now
        self.arms, self.hands, self.valid_s, self.last_source_s = [], [], 0., None
        self.phase = "prepare"

    def cancel(self, message="Calibration cancelled. Press O to begin again."):
        self.stage, self.phase, self.message = None, "cancelled", message

    def _restart(self, arm, hand, message):
        self.arms, self.hands, self.valid_s = [arm], [hand], 0.
        self.phase, self.message = "steady", message

    def update(self, now, arm, hand, new_result, source_s, reason):
        """Return (stage, result) on completion; each distinct source frame counts once."""
        if not self.active:
            return None
        if now-self.started >= self.timeout_s:
            self.cancel(f"Calibration timed out ({reason}). Fix the view, then press {'O' if self.stage == 'open' else 'C'} again.")
            return None
        remaining = self.prepare_s-(now-self.started)
        if remaining > 0:
            self.phase = "prepare"
            self.message = (f"Get ready ({remaining:.1f}s): upper arm down, forearm up, open palm facing the camera" if self.stage == "open"
                            else f"Get ready ({remaining:.1f}s): make a fist, thumb bent")
            return None
        needs_arm = self.stage == "open"
        if hand is None or (needs_arm and arm is None):
            self.phase = "paused"
            self.last_source_s = None  # A gap never counts as sampling time.
            self.message = "Paused: "+reason
            return None
        if not new_result or source_s is None or now-source_s > self.freshness_s:
            return None
        if self.last_source_s is not None and source_s <= self.last_source_s:
            return None
        self.arms.append(arm)
        self.hands.append(hand)
        if self.last_source_s is not None:
            self.valid_s += min(.1, source_s-self.last_source_s)
        self.last_source_s = source_s
        self.phase = "sampling"
        self.message = f"Hold still - {len(self.hands)} good samples"
        if len(self.hands) >= 4:
            try:
                calibration_pose(self.hands if len(self.hands) >= 8 else self.hands*2)
            except ValueError:
                self._restart(arm, hand, "Fingers moved - hold the hand pose still; restarting")
                return None
            if needs_arm:
                wrists = np.array([a["points_body_m"][2] for a in self.arms])
                if np.max(np.linalg.norm(wrists[-1]-np.median(wrists, axis=0))) > .06:
                    self._restart(arm, hand, "Arm moved - hold your arm still; restarting")
                    return None
        if len(self.hands) >= 8 and self.valid_s >= self.duration_s:
            stage = self.stage
            try:
                pose = calibration_pose(self.hands)
                result = {"pose": pose}
                if stage == "open":
                    result["anchor"] = ArmAnchor.capture(self.arms, self.hands)
                result["hand"] = median_hand(self.hands)
            except ValueError as error:
                self._restart(arm, hand, f"{error} - restarting")
                return None
            self.stage, self.phase = None, "complete"
            self.message = ("Neutral arm + open hand saved. Make a fist and press C." if stage == "open"
                            else "Closed hand saved. Checking finger travel...")
            return stage, result
        return None

    def status(self, now):
        return {"stage": self.stage, "phase": self.phase,
                "progress": min(1., self.valid_s/self.duration_s),
                "accepted_samples": len(self.hands),
                "valid_seconds": self.valid_s, "required_seconds": self.duration_s,
                "elapsed_seconds": max(0., now-self.started) if self.active else 0.,
                "message": self.message}


START_POSE_TEXT = ("Start pose: upper arm relaxed down at your side, elbow bent so the forearm points up and forward, "
                   "open hand with the palm facing the camera, fingers up.")


def start_pose_checks(arm, hand):
    """Live comparison with the recommended start pose: list of (label, ok, detail).

    These are guidance only. Calibration accepts any steady pose, and the robot
    matches whatever start pose is captured, but this one tracks best.
    """
    if arm is None:
        return [("Start pose", False, "arm not tracked")]
    points = np.asarray(arm["points_body_m"])
    upper = points[1]-points[0]
    down = float(np.degrees(np.arccos(np.clip(-upper[2]/max(np.linalg.norm(upper), 1e-9), -1, 1))))
    flexion = float(arm["elbow_flexion_deg"])
    checks = [("Upper arm down", down <= 40, f"{down:.0f} deg" + ("" if down <= 40 else ": lower your elbow")),
              ("Elbow bent", 60 <= flexion <= 145, f"{flexion:.0f} deg" + ("" if 60 <= flexion <= 145 else
                                                     (": bend elbow more" if flexion < 60 else ": open elbow a bit")))]
    if hand is None:
        return checks+[("Palm toward camera", False, "hand not tracked")]
    palm = np.asarray(arm["basis_columns_in_model_camera"]).T@np.asarray(hand["palm_basis_columns_in_model_camera"])
    palmar = palm[:, 2]*(1 if arm["side"] == "Right" else -1)   # hand_geometry normal is dorsal for a left hand
    facing = float(np.degrees(np.arccos(np.clip(palmar[0], -1, 1))))
    upward = float(np.degrees(np.arccos(np.clip(palm[2, 1], -1, 1))))
    checks.append(("Palm toward camera", facing <= 55, f"{facing:.0f} deg" + ("" if facing <= 55 else ": palm to camera")))
    checks.append(("Fingers up", upward <= 50, f"{upward:.0f} deg" + ("" if upward <= 50 else ": fingers up")))
    return checks


def finger_travel(opened, current):
    """Which controlled fingers have enough open-to-closed travel (for the C step)."""
    if opened is None or current is None:
        return {}
    return {finger: current["curl_raw"][finger]-opened["curl"][finger] >= .10 for finger in CONTROLLED}


class TeleopGate:
    """Target-stream permission. Short gaps hold the robot; longer ones latch.

    A gap shorter than grace_s sends no commands (the follower holds its pose)
    and resumes when fresh paired tracking returns. A longer gap latches
    TRACKING_LOST, which needs the operator to press Space after recovery.
    """

    def __init__(self, recovery_s=.5, grace_s=.3):
        self.recovery_s, self.grace_s = recovery_s, grace_s
        self.active = False
        self.fault = False
        self.valid_since = None
        self.invalid_since = None

    def observe(self, valid, now):
        if valid:
            self.invalid_since = None
            if self.valid_since is None:
                self.valid_since = now
            return
        self.valid_since = None
        if self.invalid_since is None:
            self.invalid_since = now
        if self.active and now-self.invalid_since > self.grace_s:
            self.active, self.fault = False, True

    @property
    def holding(self):
        return self.active and self.invalid_since is not None

    def start(self, calibrated, valid, now):
        if not calibrated or not valid or self.valid_since is None or now-self.valid_since < self.recovery_s:
            return False
        self.active, self.fault = True, False
        return True

    def stop(self):
        self.active = False

    def state(self, calibrated, calibrating=False):
        if calibrating:
            return "CALIBRATING"
        if self.fault:
            return "TRACKING_LOST"
        if not calibrated:
            return "CALIBRATION_REQUIRED"
        if self.active:
            return "TRACKING_HOLD" if self.holding else "STREAMING"
        return "READY"
