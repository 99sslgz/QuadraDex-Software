"""Simulate exactly the bounded positions carried in the control packet."""
import time
import numpy as np


class Simulation:
    def __init__(self,robot,headless=False):
        self.robot=robot
        self.target=robot.measured()
        self.expires_ns=None
        self.status="Stopped - calibrate before starting"
        self.viewer=None
        self.focus_hand=False
        self._last_focus_hand=False
        if not headless:
            import mujoco.viewer
            self.viewer=mujoco.viewer.launch_passive(robot.model,robot.data,show_left_ui=False,show_right_ui=False,key_callback=self._key)
            self.viewer.cam.lookat[:]=[.26,0,.55]
            self.viewer.cam.distance,self.viewer.cam.azimuth,self.viewer.cam.elevation=1.25,125,-20

    def _key(self,key):
        if key in (ord('F'),ord('f')):
            self.focus_hand=not self.focus_hand

    def apply(self,signal,now_ns=None):
        now_ns=time.perf_counter_ns() if now_ns is None else now_ns
        if signal.get('schema')!='quadradex.control.v1' or signal.get('clock')!='host_perf_counter_ns':
            raise ValueError("Unsupported simulator control packet")
        if signal.get('joint_names')!=self.robot.names:
            raise ValueError("Control signal joint order differs from this robot model")
        if not signal['generated_monotonic_ns']<=now_ns<=signal['expires_monotonic_ns']:
            raise ValueError("Simulator rejected an expired or future control signal")
        target=np.asarray(signal['positions_rad'],float)
        if target.shape!=(23,) or not np.isfinite(target).all() or np.any(target<self.robot.lower) or np.any(target>self.robot.upper):
            raise ValueError("Simulator rejected an invalid joint target")
        self.target,self.expires_ns=target.copy(),signal['expires_monotonic_ns']
        self.status="Following generated arm + LEAP targets"

    def hold(self,reason="Stopped"):
        if self.expires_ns is not None:
            self.target=self.robot.measured()
        self.expires_ns=None
        self.status=reason

    def step(self,dt,now_ns=None):
        now_ns=time.perf_counter_ns() if now_ns is None else now_ns
        if self.expires_ns is not None and now_ns>self.expires_ns:
            self.hold("Control signal expired - holding simulated pose")
        if self.viewer is not None:
            with self.viewer.lock():
                self.robot.advance(self.target,dt)
                if self.focus_hand!=self._last_focus_hand:
                    self.viewer.cam.distance=.38 if self.focus_hand else 1.25
                    if not self.focus_hand:
                        self.viewer.cam.lookat[:]=[.26,0,.55]
                    self._last_focus_hand=self.focus_hand
                if self.focus_hand:
                    self.viewer.cam.lookat[:]=self.robot.data.body('leap_palm_lower').xpos
            self.viewer.sync()
        else:
            self.robot.advance(self.target,dt)

    def is_running(self):
        return self.viewer is None or self.viewer.is_running()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
