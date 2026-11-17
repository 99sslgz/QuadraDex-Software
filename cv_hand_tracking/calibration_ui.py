"""Guided calibration with preparation, accepted-sample progress and pause reasons."""
from hand_geometry import CONTROLLED, calibration_pose


class CalibrationCapture:
    def __init__(self, duration_s=1.25, prepare_s=1.0, timeout_s=25.):
        self.duration_s, self.prepare_s, self.timeout_s = duration_s, prepare_s, timeout_s
        self.stage = None
        self.started = 0.
        self.samples = []
        self.valid_s = 0.
        self.last_source_s = None
        self.message = "Step 1 of 2: open your hand and press O"
        self.phase = "idle"

    @property
    def active(self):
        return self.stage is not None

    def start(self, stage, now):
        self.stage, self.started = stage, now
        self.samples, self.valid_s, self.last_source_s = [], 0., None
        self.phase = "prepare"
        self.message = "Open and spread your fingers" if stage == "open" else "Close your fingers and bend your thumb"

    def cancel(self, message="Calibration cancelled. Press O to begin again."):
        self.stage, self.phase, self.message = None, "cancelled", message

    def update(self, now, measurement, new_result, source_s, reason):
        """Return (stage, pose) on completion; each distinct source may count once."""
        if not self.active:
            return None
        if now - self.started >= self.timeout_s:
            self.cancel("Calibration timed out. Adjust the camera, then retry this pose.")
            return None
        remaining = self.prepare_s - (now - self.started)
        if remaining > 0:
            self.phase = "prepare"
            self.message = f"Get ready: {remaining:.1f}s - hold the {self.stage.upper()} pose"
            return None
        if measurement is None:
            self.phase = "paused"
            self.last_source_s = None  # A gap must never count as sampling time.
            self.message = "Paused: " + reason
            return None
        if not new_result or source_s is None or now - source_s > .15:
            if self.last_source_s is None or now - self.last_source_s > .15:
                self.phase, self.message = "waiting", "Waiting for a fresh hand measurement..."
            return None
        if self.last_source_s is not None and source_s <= self.last_source_s:
            return None
        self.samples.append(measurement)
        if self.last_source_s is not None:
            self.valid_s += min(.1, source_s - self.last_source_s)
        self.last_source_s = source_s
        self.phase = "sampling"
        self.message = f"Hold still - {len(self.samples)} accepted samples"
        if len(self.samples) >= 8:
            try:
                pose = calibration_pose(self.samples)
            except ValueError:
                # Restart the stable window automatically, without exiting calibration.
                self.samples, self.valid_s = [measurement], 0.
                self.phase = "steady"
                self.message = "Movement detected - hold this pose still; restarting the stable sample"
                return None
            if self.valid_s >= self.duration_s:
                stage = self.stage
                self.stage, self.phase = None, "complete"
                self.message = "Open pose saved. Close your hand and press C." if stage == "open" else "Both poses captured. Checking finger ranges..."
                return stage, pose
        return None

    def status(self, now):
        return {"stage": self.stage, "phase": self.phase,
                "progress": min(1., self.valid_s / self.duration_s),
                "accepted_samples": len(self.samples),
                "valid_seconds": self.valid_s, "required_seconds": self.duration_s,
                "elapsed_seconds": max(0., now - self.started) if self.active else 0.,
                "message": self.message}


def finger_ranges(opened, current):
    """Show which controlled fingers have enough measured open-to-closed travel."""
    if opened is None:
        return {}
    return {finger: {"span": current["curl_raw"][finger] - opened["curl"][finger],
                     "ready": current["curl_raw"][finger] - opened["curl"][finger] >= .10}
            for finger in CONTROLLED}
