"""Simulate exactly the bounded positions carried in the control packet.

The interactive window shows the follower station, ghost markers for the
tracked targets (orange: your arm posture at robot scale, green sphere/axes:
commanded wrist pose, green dots: fingertip targets) and a status overlay.
"""
import time
import numpy as np
import mujoco

from teleoperation.robot_model import default_camera
from teleoperation.visuals import draw_overlay


class Simulation:
    def __init__(self, robot, headless=False):
        self.robot = robot
        self.target = robot.measured()
        self.expires_ns = None
        self.status = "Stopped - calibrate before starting"
        self.viewer = None
        self.overlay = None
        self.hud = {}
        self.focus_hand = False
        self._last_focus_hand = False
        self._reset_objects = False
        if not headless:
            import mujoco.viewer
            self.viewer = mujoco.viewer.launch_passive(robot.model, robot.data, show_left_ui=False, show_right_ui=False,
                                                       key_callback=self._key)
            with self.viewer.lock():
                default_camera(self.viewer.cam, robot.config)
                self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0

    def _key(self, key):
        # Function keys avoid the viewer's own single-letter shortcuts.
        if key == 297:      # F8: follow the hand / overview
            self.focus_hand = not self.focus_hand
        elif key == 298:    # F9: put the objects back
            self._reset_objects = True

    def apply(self, signal, now_ns=None):
        now_ns = time.perf_counter_ns() if now_ns is None else now_ns
        if signal.get('schema') != 'quadradex.control.v1' or signal.get('clock') != 'host_perf_counter_ns':
            raise ValueError("Unsupported simulator control packet")
        if signal.get('joint_names') != self.robot.names:
            raise ValueError("Control signal joint order differs from this robot model")
        if not signal['generated_monotonic_ns'] <= now_ns <= signal['expires_monotonic_ns']:
            raise ValueError("Simulator rejected an expired or future control signal")
        target = np.asarray(signal['positions_rad'], float)
        if target.shape != (23,) or not np.isfinite(target).all() or np.any(target < self.robot.lower) or np.any(target > self.robot.upper):
            raise ValueError("Simulator rejected an invalid joint target")
        self.target, self.expires_ns = target.copy(), signal['expires_monotonic_ns']
        self.status = "Following generated arm + LEAP targets"

    def hold(self, reason="Stopped"):
        if self.expires_ns is not None:
            # Hold where the robot is, clamped into the model's joint ranges.
            self.target = np.clip(self.robot.measured(), self.robot.lower, self.robot.upper)
        self.expires_ns = None
        self.status = reason

    def request_object_reset(self):
        self._reset_objects = True

    def set_overlay(self, overlay):
        """Target ghosts to draw (or None to hide them)."""
        self.overlay = overlay

    def set_hud(self, title, lines):
        self.hud = {"title": title, "lines": lines}

    def step(self, dt, now_ns=None):
        now_ns = time.perf_counter_ns() if now_ns is None else now_ns
        if self.expires_ns is not None and now_ns > self.expires_ns:
            self.hold("Control signal expired - holding simulated pose")
        if self.viewer is None:
            if self._reset_objects:
                self.robot.reset_objects()
                self._reset_objects = False
            self.robot.advance(self.target, dt)
            return
        with self.viewer.lock():
            if self._reset_objects:
                self.robot.reset_objects()
                self._reset_objects = False
            self.robot.advance(self.target, dt)
            if self.focus_hand != self._last_focus_hand:
                default_camera(self.viewer.cam, self.robot.config, close=self.focus_hand)
                self._last_focus_hand = self.focus_hand
            if self.focus_hand:
                self.viewer.cam.lookat[:] = self.robot.data.body('leap_palm_lower').xpos
            scene = self.viewer.user_scn
            scene.ngeom = 0
            draw_overlay(scene, self.overlay)
        if self.hud:
            self.viewer.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPLEFT, self.hud["title"], self.status),
                (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT, "\n".join(k for k, _ in self.hud["lines"]),
                 "\n".join(v for _, v in self.hud["lines"])),
            ])
        self.viewer.sync()

    def is_running(self):
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
