"""Run synchronized full-arm + finger tracking and the LEAP V1 motion simulator.

python -m teleoperation.run              # camera picker + tracking + simulation
python -m teleoperation.run --demo       # synthetic motion, no camera required

The session only ends when the operator asks (Q / Esc, or closing a window).
Limits, tracking loss, camera failures and internal errors hold the simulated
robot, explain what happened on screen and keep the program running.
"""
from __future__ import annotations
import argparse
from collections import deque
from pathlib import Path
import socket
import sys
import threading
import time
import traceback
import uuid

import cv2
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"cv_hand_tracking"))
from track_hand import LatestCamera, Output
from camera_picker import choose_camera, discover_cameras, CameraChoice
from hand_geometry import features, make_calibration
from teleoperation.calibration import (START_POSE_TEXT, PairedCalibration, TeleopGate, finger_travel,
                                       start_pose_checks)
from teleoperation.robot_model import RobotModel
from teleoperation.retarget import Retargeter, SignalGenerator
from teleoperation.simulator import Simulation
from teleoperation.stereo import (USE_MIN_CONFIDENCE, StereoCalibration, StereoCollector, StereoFusion, calibrate,
                                  pair_observations)
from teleoperation.vision import Observation, WholeArmVision, ensure_models
from track_hand import Frame
from teleoperation import synthetic_hand
from teleoperation import ui as panel

WINDOW = "QuadraDex | Full arm + LEAP control"
START_NOTE = ("Sit or stand 1.2-2 m from the camera: head, both shoulders and the whole selected arm in view. "
              + START_POSE_TEXT+" Press O.")
ALIGN_DONE_RAD = .08
DEVICE_RETRY_S = 3.


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cameras', type=int, choices=(1, 2),
                        help='One camera, or two cameras (front + side) fused by triangulation; asked at start-up if omitted')
    parser.add_argument('--camera', type=int)
    parser.add_argument('--camera2', type=int, help='Second camera index in two-camera mode')
    parser.add_argument('--mirrored2', action='store_true', help='The second camera delivers a mirrored image')
    parser.add_argument('--stereo-file', type=Path, default=ROOT/'stereo_calibration.json',
                        help='Saved two-camera setup, reused while the same cameras and resolution are used')
    parser.add_argument('--list-cameras', action='store_true')
    parser.add_argument('--backend', choices=('auto', 'msmf', 'dshow'), default='auto')
    parser.add_argument('--side', choices=('right', 'left'), default='right', help='Operator arm to track')
    parser.add_argument('--mirrored', '--swap-handedness', dest='mirrored', action='store_true',
                        help='The camera driver delivers a mirrored image (selected arm highlights the wrong side)')
    parser.add_argument('--pose-model', choices=('lite', 'full', 'heavy'), default='full')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=720)
    parser.add_argument('--camera-fps', type=float, default=30)
    parser.add_argument('--min-visibility', type=float, default=.5)
    parser.add_argument('--max-age-ms', type=float, help='Oldest tracking result used for control (default 120 ms, 160 ms with two cameras)')
    parser.add_argument('--loss-grace-ms', type=float, default=500,
                        help='Tracking gaps shorter than this hold the robot instead of latching TRACKING_LOST')
    parser.add_argument('--part-hold-ms', type=float, default=800,
                        help='How long a briefly missing arm or hand is bridged with its last measurement')
    parser.add_argument('--output-hz', type=float, default=25)
    parser.add_argument('--arm-config', type=Path, default=ROOT/'demo_arm.json')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--demo', action='store_true', help='Synthetic arm/finger motion, clearly marked in exported packets')
    parser.add_argument('--snapshot', type=Path, help='Save the simulated robot view on exit; no camera image is saved')
    parser.add_argument('--download-models', action='store_true')
    parser.add_argument('--log', type=Path, help='New JSONL log, never overwrites an existing file')
    parser.add_argument('--udp', help='Optional same-host JSON output, e.g. 127.0.0.1:5006')
    args = parser.parse_args(argv)
    for key in ('width', 'height', 'camera_fps', 'max_age_ms', 'output_hz', 'part_hold_ms'):
        if getattr(args, key) is None:
            continue
        if not np.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'{key} must be positive and finite')
    if not np.isfinite(args.loss_grace_ms) or not 0 <= args.loss_grace_ms <= 1000:
        parser.error('loss-grace-ms must be between 0 and 1000')
    if args.camera is not None and args.camera < 0:
        parser.error('Camera index must be nonnegative')
    if not 0 < args.min_visibility <= 1:
        parser.error('Visibility threshold must be in (0,1]')
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error('Duration must be positive and finite')
    if args.headless and args.duration is None and not (args.download_models or args.list_cameras):
        parser.error('Use --duration with --headless')
    if args.udp:
        try:
            host, port = args.udp.rsplit(':', 1)
            if socket.gethostbyname(host) != '127.0.0.1' or not 1 <= int(port) <= 65535:
                raise ValueError()
            args.udp = ('127.0.0.1', int(port))
        except (OSError, ValueError):
            parser.error('This prototype exports to 127.0.0.1:PORT on the same computer')
    return args


def json_ready(value):
    """Convert NumPy scalars/arrays (np.bool_, np.float32, ndarray, ...) nested in a packet to plain Python."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class PacketOutput(Output):
    """Output that accepts NumPy values and never stops the session on a write problem."""

    def emit(self, packet):
        try:
            super().emit(json_ready(packet))
        except (TypeError, ValueError) as error:
            self.log_error = self.log_error or f"Packet not exported: {error}"


def envelope(session, sequence, state, reason, signal=None, observation=None, metrics=None, synthetic=False):
    return {'schema': 'quadradex.teleop.v1', 'session_id': session, 'sequence': sequence,
            'published_monotonic_ns': time.perf_counter_ns(), 'clock': 'host_perf_counter_ns',
            'state': state, 'reason': reason, 'synthetic_demo': synthetic, 'hardware_enabled': False,
            'arm_accuracy': 'not_independently_validated', 'control_signal': signal,
            'source_frame_id': observation.frame.index if observation else None,
            'arm_observation': observation.arm if observation else None,
            'hand_measurement': observation.hand if observation else None, 'fit_metrics': metrics}


def hud_lines(state, fit, extra=()):
    lines = [("State", state.replace('_', ' '))]
    if fit:
        lines += [("Wrist IK", f"{fit['arm_ik_error_mm']:.0f} mm, {fit['wrist_orientation_error_deg']:.0f} deg"),
                  ("Thumb fit", f"{fit['mean_fingertip_fit_error_mm']:.0f} mm")]
        pinch = fit.get('pinch') or {}
        if pinch.get('finger'):
            lines.append(("Pinch", f"thumb + {pinch['finger']} ({100*pinch['weight']:.0f}%, gap {pinch['robot_gap_mm']:.0f} mm)"))
        if fit.get('limit_reasons'):
            lines.append(("LIMIT", "; ".join(fit['limit_reasons'])))
        if fit.get('near_limits'):
            lines.append(("Near limit", ", ".join(fit['near_limits'])[:80]))
        if fit.get('guidance'):
            lines.append(("Do this", fit['guidance'][:80]))
    lines += list(extra)
    lines += [("Legend", "orange = your arm, green = wrist / thumb targets"),
              ("Views", "[ ] cameras, F8 follow hand, F9 reset objects")]
    return lines


# ---------------------------------------------------------------------- demo
def demo_hand(t):
    """Synthetic operator hand cycling through open, pinches, a fist and single-finger curls."""
    phase = (t/2.5) % 6
    k = int(phase)
    s = .5*(1-np.cos(np.pi*min(1., 2*(phase-k))))  # ease in, hold
    open_hand = synthetic_hand.hand_landmarks()
    if k == 1:
        target, _ = synthetic_hand.pinch_landmarks('index')
    elif k == 2:
        target, _ = synthetic_hand.pinch_landmarks('middle')
    elif k == 3:
        target = synthetic_hand.hand_landmarks({f: (0, 80, 95, 55) for f in synthetic_hand.MCP}, synthetic_hand.FIST_THUMB)
    elif k == 4:
        target = synthetic_hand.hand_landmarks({'index': (0, 0, 0, 0), 'middle': (0, 75, 90, 50), 'ring': (0, 80, 95, 55)},
                                               synthetic_hand.FIST_THUMB)
    elif k == 5:
        target = synthetic_hand.hand_landmarks({'index': (14, 0, 0, 0), 'ring': (-12, 0, 0, 0)})
    else:
        target = open_hand
    local = (1-s)*open_hand+s*target
    return features(synthetic_hand.to_camera(local, synthetic_hand.PALM_TOWARD_CAMERA))


def run_demo(args, robot, simulation, output):
    """Synthetic wrist path through the arm IK, and a synthetic human hand through the real hand mapping."""
    generator = SignalGenerator(robot, args.max_age_ms or 120.)
    solver = Retargeter(robot)
    open_hand = demo_hand(0.)
    solver.human_open = open_hand
    solver.thumb_scale = solver.leap_thumb_reach/float(np.linalg.norm(
        np.subtract(open_hand['landmarks_palm_model_m'][4], open_hand['landmarks_palm_model_m'][1])))
    solver.calibrate_closed(features(synthetic_hand.to_camera(
        synthetic_hand.hand_landmarks({f: (0, 80, 95, 55) for f in synthetic_hand.MCP}, synthetic_hand.FIST_THUMB),
        synthetic_hand.PALM_TOWARD_CAMERA)))
    session = str(uuid.uuid4())
    start = previous = time.perf_counter()
    sequence = 0
    print('SYNTHETIC DEMO: simulated arm + LEAP V1. No camera or physical actuators are used.', flush=True)
    q = robot.home.copy()
    while simulation.is_running():
        now = time.perf_counter()
        t = now-start
        if args.duration is not None and t >= args.duration:
            break
        try:
            target = solver.home_tool+np.array([.10*np.sin(t*.55), .20*np.sin(t*.37), .12*np.sin(t*.71)+.04])
            angles = np.array([.35*np.sin(t*.5), .30*np.sin(t*.43), .25*np.sin(t*.61)])
            rotation = np.empty(9)
            quat = np.empty(4)
            mujoco.mju_axisAngle2Quat(quat, angles/max(np.linalg.norm(angles), 1e-9), np.linalg.norm(angles))
            mujoco.mju_quat2Mat(rotation, quat)
            orientation = rotation.reshape(3, 3)@solver.home_rotation
            q, arm_error, rotation_error = solver.solve_arm(target, orientation, seed=q, iterations=25)
            q, hand_fit = solver.solve_hand(demo_hand(t), q)
            signal = generator.generate(q, time.perf_counter_ns())
            simulation.apply(signal)
            data = robot.forward(q)
            root = data.body('leap_palm_lower')
            simulation.set_overlay({"wrist": target, "wrist_rotation": orientation,
                                    "tips": [root.xpos+root.xmat.reshape(3, 3)@hand_fit["thumb_goal_palm"]]})
            fit = {'arm_ik_error_mm': arm_error*1000, 'wrist_orientation_error_deg': np.degrees(rotation_error),
                   'mean_fingertip_fit_error_mm': hand_fit['thumb_fit_mm'], 'limit_reasons': [],
                   'pinch': {'finger': hand_fit['pinch_finger'], 'weight': hand_fit['pinch_weight'],
                             'robot_gap_mm': hand_fit['pinch_gap_mm']}}
            simulation.set_hud("QuadraDex follower - SYNTHETIC DEMO", hud_lines('SYNTHETIC_DEMO', fit))
            simulation.step(min(.05, now-previous))
            sequence += 1
            output.emit(envelope(session, sequence, 'SYNTHETIC_DEMO', 'Generated demo motion; not a measured operator pose', signal, synthetic=True))
        except Exception as error:  # the demo never stops on its own either
            print(f'Demo step skipped: {type(error).__name__}: {error}', file=sys.stderr, flush=True)
            simulation.hold('Demo step skipped')
        previous = now
        time.sleep(max(0, 1/args.output_hz-(time.perf_counter()-now)))
    output.emit(envelope(session, sequence+1, 'STOPPED', 'Demo finished', synthetic=True))


# ---------------------------------------------------------------------- live
class LiveSession:
    """Camera tracking (one or two cameras) -> retargeting -> simulation, with every failure handled in place."""

    def __init__(self, args, robot, simulation, output):
        self.args, self.robot, self.simulation, self.output = args, robot, simulation, output
        self.mode = args.cameras
        self.views = []                       # [{'choice', 'camera', 'worker'}]
        self.known_choices = []
        self.mirrored = [bool(args.mirrored), bool(args.mirrored2)]
        self.clicked = []
        self.retarget = Retargeter(robot)
        self.generator = SignalGenerator(robot, self.max_age_ms)
        self.session, self.sequence = str(uuid.uuid4()), 0
        self.started = self.last_tick = self.last_publish = self.last_console = time.perf_counter()
        self.device_error = None
        self.device_retry_at = 0.
        self.fault = None
        self.fault_logged = set()
        self.log_warning = None
        self.observation = None
        self.state, self.reason = 'STOPPED', 'Starting camera and models...'
        # Two-camera state.
        self.stereo = self.fusion = self.collector = None
        self.stereo_job = None
        self.stereo_message = ''
        self.stereo_warning = None
        self.pair_key, self.pair_index, self.composite = None, 0, None
        self.reset_calibration()

    @property
    def max_age_ms(self):
        if self.args.max_age_ms is not None:
            return self.args.max_age_ms
        return 160. if self.mode == 2 else 120.

    @property
    def worker(self):
        return self.views[0]['worker'] if self.views else None

    @property
    def choice(self):
        return self.views[0]['choice'] if self.views else (self.known_choices[0] if self.known_choices else None)

    # ---------------------------------------------------------------- state helpers
    def reset_calibration(self):
        self.gate = TeleopGate(grace_s=self.args.loss_grace_ms/1000)
        self.calibration = PairedCalibration()
        self.anchor = self.hand_calibration = self.opened = None
        self.reference = {}
        self.start_wrist = None
        self.active_signal = self.fit = None
        self.control_reason = ''
        self.aligning = False
        self.note = START_NOTE
        self.last_result = self.last_solved = -1
        self.times = deque()
        self.pose_checks = []
        self.last_arm = self.last_hand = None
        self.partial = ''

    @property
    def calibrated(self):
        return self.hand_calibration is not None and self.anchor is not None

    def hold(self, reason):
        self.active_signal = None
        self.simulation.hold(reason)

    def emit(self, state, reason, **extra):
        self.sequence += 1
        packet = envelope(self.session, self.sequence, state, reason, self.active_signal, self.observation,
                          self.fit if self.gate.active else None)
        packet.update(extra)
        self.output.emit(packet)

    # ---------------------------------------------------------------- devices
    def choose_mode(self):
        if self.mode in (1, 2):
            return self.mode
        if self.args.headless:
            return 2 if self.args.camera2 is not None else 1
        return panel.choose_mode()

    def camera_names(self):
        return [f"{v['choice'].name} [{v['choice'].index}]" for v in self.views]

    def make_worker(self, k, camera):
        return WholeArmVision(camera, self.args.side, self.mirrored[k], self.args.min_visibility, self.args.pose_model)

    def attach(self, specs):
        """Use already opened cameras: [(choice, camera), ...]; starts one tracking worker per camera."""
        self.views = [{'choice': choice, 'camera': camera, 'worker': None} for choice, camera in specs]
        self.known_choices = [choice for choice, _ in specs]
        for k, view in enumerate(self.views):
            view['worker'] = self.make_worker(k, view['camera'])

    def open_views(self, pick=True):
        """Open the camera(s): the picker when interactive and not yet chosen, else the known/CLI indices."""
        args = self.args
        specs = []
        try:
            for k in range(self.mode):
                index = args.camera if k == 0 else args.camera2
                if k < len(self.known_choices):
                    choice = self.known_choices[k]
                elif index is None and pick and not args.headless:
                    title = ("Choose your tracking camera" if self.mode == 1 else
                             "Camera 1 of 2: FRONT camera (keep it upright)" if k == 0 else
                             "Camera 2 of 2: SIDE camera, 45-90 degrees to one side")
                    picked = choose_camera(LatestCamera, args, title, exclude=[c.index for c, _ in specs])
                    if picked is None:
                        raise LookupError('No camera selected')
                    specs.append(picked)
                    continue
                else:
                    index = index if index is not None else k
                    choice = CameraChoice(index, args.backend, f'Camera {index}')
                specs.append((choice, LatestCamera(choice.index, args.width, args.height, args.camera_fps, choice.backend)))
        except Exception:
            for _, camera in specs:
                camera.close()
            raise
        self.attach(specs)

    def close_devices(self):
        for view in self.views:
            for device in (view['worker'], view['camera']):
                if device is not None:
                    try:
                        device.close()
                    except Exception as error:
                        print(f'Closing a device failed: {error}', file=sys.stderr, flush=True)
        self.views = []
        self.composite = self.pair_key = None

    def device_failed(self, message, now):
        """Camera or inference failure: hold, explain, and retry automatically."""
        if self.device_error is None:
            print(f'Camera/tracking problem: {message}. Retrying every {DEVICE_RETRY_S:.0f} s.', file=sys.stderr, flush=True)
        self.device_error = message
        self.device_retry_at = now+DEVICE_RETRY_S
        self.gate.stop()
        self.hold('Camera problem - holding pose')
        self.close_devices()

    def retry_devices(self, now):
        if now < self.device_retry_at:
            return
        try:
            self.open_views(pick=False)
            self.device_error = None
            self.started = now
            self.note = 'Camera restored. Re-check the view; Space resumes when tracking is steady.'
        except Exception as error:
            self.close_devices()
            self.device_error = str(error)
            self.device_retry_at = now+DEVICE_RETRY_S

    # ---------------------------------------------------------------- two-camera setup
    def stereo_path(self):
        return self.args.stereo_file

    def begin_stereo(self, observations):
        """Called once both cameras deliver images: reuse a matching saved calibration or start the setup."""
        sizes = [list(o.image_size) for o in observations]
        try:
            calibration = StereoCalibration.load(self.stereo_path())
            if calibration.matches(self.camera_names(), sizes):
                self.set_stereo(calibration, 'Loaded the saved two-camera setup')
                return
        except (OSError, ValueError, TypeError, KeyError):
            pass
        self.start_stereo_setup()

    def start_stereo_setup(self):
        self.stereo = self.fusion = None
        self.stereo_job = None
        self.collector = StereoCollector()
        self.stereo_warning = None
        self.reset_calibration()   # coordinates change with a new setup
        self.note = 'Two-camera setup: ' + self.collector.message

    def set_stereo(self, calibration, message):
        self.stereo = calibration
        self.fusion = StereoFusion(calibration, self.args.side, USE_MIN_CONFIDENCE)
        self.collector = None
        self.stereo_message = f'{message}: quality {calibration.quality}, {calibration.rms_px:.1f} px, cameras {calibration.angle_deg:.0f} deg apart.'
        advice = ' Reposition the cameras (45-90 deg apart, both seeing your whole arm) and press K.' if calibration.quality == 'poor' else ''
        self.stereo_warning = ('Two-camera setup is poor.'+advice) if advice else None
        self.note = self.stereo_message+' '+START_NOTE

    def stereo_setup_tick(self, o1, o2, skew, now):
        if self.stereo_job is not None:
            job = self.stereo_job
            if job['thread'].is_alive():
                self.stereo_message = 'Computing the two-camera geometry...'
                return
            self.stereo_job = None
            if job.get('error'):
                self.stereo_message = f"Setup failed: {job['error']}. Collecting again."
                print(self.stereo_message, file=sys.stderr, flush=True)
                self.collector = StereoCollector()
                return
            calibration = job['result']
            try:
                calibration.save(self.stereo_path())
            except OSError as error:
                print(f'Could not save the two-camera setup: {error}', file=sys.stderr, flush=True)
            self.set_stereo(calibration, 'Two-camera setup done')
            return
        if self.collector is None:
            self.collector = StereoCollector()
        self.collector.update(o1, o2, skew, now)
        self.stereo_message = self.collector.message
        if self.collector.complete:
            samples, names = list(self.collector.samples), self.camera_names()
            sizes = [list(o1.image_size), list(o2.image_size)]
            job = {}

            def solve():
                try:
                    job['result'] = calibrate(samples, names, sizes)
                except Exception as error:
                    job['error'] = str(error)
            job['thread'] = threading.Thread(target=solve, daemon=True, name='stereo-calibration')
            job['thread'].start()
            self.stereo_job = job

    def stereo_observation(self, o1, o2):
        """Fuse a time-paired observation from each camera into one composite observation."""
        key = (o1.frame.index if o1 else -1, o2.frame.index if o2 else -1)
        if key == self.pair_key and self.composite is not None:
            return self.composite
        self.pair_key = key
        self.pair_index += 1
        available = [o for o in (o1, o2) if o is not None]
        received_ns = min(o.frame.received_ns for o in available)
        source_s = max(o.frame.received_ns for o in available)/1e9
        result = self.fusion.fuse(o1, o2, source_s)
        reason = ("Fused shoulder / elbow / wrist / fingers from two cameras" if result.arm and result.hand else
                  (result.arm_reason or result.hand_reason))
        hand_image = next((o.hand_image for o in available if o.hand_image is not None), None)
        timing = {name: max(o.timing_ms.get(name, 0) for o in available) for name in ('pose', 'hand')}
        frame = Frame(self.pair_index, available[0].frame.image, received_ns, 0.)
        live = result.info.get('median_reprojection_px')
        if live is not None and len(self.fusion.recent_errors) >= 60 and live > max(10., 3*self.stereo.rms_px):
            self.stereo_warning = f'Cameras may have moved (views disagree by {live:.0f} px). Press K to redo the two-camera setup.'
        self.composite = Observation(frame, time.perf_counter_ns(), result.arm, result.hand, hand_image,
                                     available[0].pose_image, reason, None, result.parts, result.arm_reason,
                                     result.hand_reason, None, timing, views=(o1, o2), fusion=result.info)
        return self.composite

    # ---------------------------------------------------------------- main loop
    def run(self):
        args = self.args
        self.mode = self.choose_mode()
        if self.mode is None:
            return
        self.generator = SignalGenerator(self.robot, self.max_age_ms)
        try:
            self.open_views()
        except LookupError:
            return
        except Exception as error:
            self.device_failed(str(error), time.perf_counter())
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(WINDOW, panel.WIDTH, panel.HEIGHT)
            cv2.setMouseCallback(WINDOW, self.mouse)
        self.started = self.last_tick = self.last_publish = self.last_console = time.perf_counter()
        try:
            while self.simulation.is_running():
                if args.duration is not None and time.perf_counter()-self.started >= args.duration:
                    break
                if self.step():
                    break
                time.sleep(.002)
        finally:
            try:
                self.active_signal = None
                self.emit('STOPPED', 'Producer stopped')
                self.simulation.hold('Producer stopped')
            finally:
                self.close_devices()
                if not args.headless:
                    cv2.destroyAllWindows()

    def step(self):
        """One protected cycle: any unexpected error is handled in place. True = operator quit."""
        try:
            return self.tick()
        except Exception as error:
            return self.on_fault(error)

    def mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            key = panel.button_at(x, y)
            if key is not None:
                self.clicked.append(key)

    def read_key(self):
        """Window key or button press; -1 if none. Returns 'quit' if the window was closed."""
        if self.args.headless:
            return -1
        key = cv2.waitKey(1) & 255
        if self.clicked:
            key = self.clicked.pop(0)
        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            return 'quit'
        return key

    def on_fault(self, error):
        """Unexpected error inside one cycle: hold the robot, report, keep running."""
        message = f'{type(error).__name__}: {error}'
        if message not in self.fault_logged:
            self.fault_logged.add(message)
            print('Recovered from an internal error (the session continues):', file=sys.stderr)
            traceback.print_exception(error, file=sys.stderr)
        self.fault = message
        try:
            self.gate.stop()
            self.calibration.cancel('Calibration interrupted by an internal error. Press O to begin again.') if self.calibration.active else None
            self.hold('Internal error - holding pose')
            self.simulation.set_hud("QuadraDex follower - simulation only", hud_lines('FAULT', None, [("Error", message[:80])]))
            self.simulation.step(min(.05, time.perf_counter()-self.last_tick))
        except Exception:
            pass
        self.last_tick = time.perf_counter()
        # Keep the operator in control even if drawing itself is failing.
        try:
            if not self.args.headless:
                screen = np.full((panel.HEIGHT, panel.WIDTH, 3), panel.BG, np.uint8)
                for i, line in enumerate(["INTERNAL ERROR - robot holding its pose, program still running", message[:110],
                                          "Space: resume when tracking is valid   R: reset   Q: quit"]):
                    cv2.putText(screen, line, (30, 60+40*i), panel.FONT, .7, panel.BAD if i == 0 else panel.INK, 2 if i == 0 else 1, cv2.LINE_AA)
                cv2.imshow(WINDOW, screen)
            key = self.read_key()
        except Exception:
            key = -1
        if key == 'quit' or key in (27, ord('q')):
            return True
        if key in (ord(' '), ord('r')):
            self.fault = None
            if key == ord('r'):
                self.reset_calibration()
        return False

    def tick(self):
        """One cycle. Returns True when the operator asked to quit."""
        now = time.perf_counter()
        args = self.args
        if self.device_error is None:
            for view in self.views:
                if view['worker'].error or view['camera'].error:
                    self.device_failed(view['worker'].error or view['camera'].error, now)
                    break
            if not self.views and self.device_error is None:
                self.device_failed('No camera open', now)
        if self.device_error is not None:
            self.retry_devices(now)
            return self.idle_screen(now, 'CAMERA_ERROR', f'{self.device_error}. Retrying automatically; V chooses other cameras.')
        if self.mode == 2:
            o1, o2, skew = pair_observations(self.views[0]['worker'].history(), self.views[1]['worker'].history())
            if o1 is None and o2 is None:
                return self.waiting(now)
            if self.stereo is None:
                if self.collector is None and self.stereo_job is None:
                    first = [view['worker'].get() for view in self.views]
                    if any(o is None for o in first):
                        return self.waiting(now)
                    self.begin_stereo(first)
                if self.stereo is None:
                    return self.setup_screen(now, o1, o2, skew)
            observation = self.stereo_observation(o1, o2)
        else:
            observation = self.worker.get()
            if observation is None:
                return self.waiting(now)
        self.observation = observation
        self.track(observation, now)
        self.publish_state(now)
        key = -1
        if not args.headless:
            cv2.imshow(WINDOW, panel.render(observation, self.ui_state(now)))
            key = self.read_key()
            if key == 'quit':
                return True
        if self.handle_key(key):
            return True
        self.publish(now)
        if self.output.log_error and self.log_warning is None:
            self.log_warning = f'Logging stopped: {self.output.log_error}'
            print(self.log_warning, file=sys.stderr, flush=True)
        if args.headless and now-self.last_console >= 1:
            print(f'{self.state}: {self.reason}; fresh paired results {len(self.times)/min(2, max(.01, now-self.started)):.1f}/s', flush=True)
            self.last_console = now
        return False

    def waiting(self, now):
        message = ('Starting camera and models...' if now-self.started < 20 else
                   'Still no body/hand result: check the camera view and lighting, or press V to pick another camera.')
        return self.idle_screen(now, 'STOPPED', message)

    def setup_screen(self, now, o1, o2, skew):
        """Two-camera setup: show both views, collect correspondences, solve in the background."""
        self.stereo_setup_tick(o1, o2, skew, now)
        self.state, self.reason = 'STEREO_SETUP', self.stereo_message
        self.observation = None
        parts = []
        for k, o in enumerate((o1, o2), start=1):
            seen = (o is not None and o.pose_confidence is not None and min(o.pose_confidence[11], o.pose_confidence[12]) >= .75)
            parts.append((f"Camera {k} sees you", seen, None, "" if seen else "shoulders not clearly visible to this camera"))
        hand_image = next((o.hand_image for o in (o1, o2) if o is not None and o.hand_image is not None), None)
        composite = Observation(Frame(0, (o1 or o2).frame.image, (o1 or o2).frame.received_ns, 0.), time.perf_counter_ns(),
                                None, None, hand_image, None, self.stereo_message, None, parts, 'two-camera setup in progress', '',
                                None, {}, views=(o1, o2))
        self.simulation.set_hud("QuadraDex follower - simulation only", hud_lines('STEREO_SETUP', None, [("Status", self.stereo_message[:80])]))
        self.simulation.step(min(.04, now-self.last_tick))
        self.last_tick = now
        if now-self.last_publish >= 1/self.args.output_hz:
            self.active_signal = None
            self.emit('STEREO_SETUP', self.stereo_message, camera_mode=2,
                      stereo={'progress': self.collector.progress if self.collector else 1.})
            self.last_publish = now
        if self.args.headless:
            time.sleep(.005)
            return False
        cv2.imshow(WINDOW, panel.render(composite, self.ui_state(now)))
        key = self.read_key()
        if key == 'quit' or key in (27, ord('q')):
            return True
        if key in (ord('v'), ord('m'), ord('n'), ord('h')):
            self.handle_key(key)
        return False

    def idle_screen(self, now, state, message):
        self.state, self.reason = state, message
        self.simulation.set_hud("QuadraDex follower - simulation only", hud_lines(state, None, [("Status", message[:80])]))
        self.simulation.step(min(.04, now-self.last_tick))
        self.last_tick = now
        if now-self.last_publish >= 1/self.args.output_hz:
            # Consumers see camera problems and start-up too; no command is ever sent from here.
            self.active_signal = None
            self.emit(state, message, fault=self.fault, camera_error=self.device_error)
            self.last_publish = now
        if self.args.headless:
            time.sleep(.005)
            return False
        cv2.imshow(WINDOW, panel.render(None, {**self.ui_state(now), 'state': state, 'reason': message}))
        key = self.read_key()
        if key == 'quit' or key in (27, ord('q')):
            return True
        if key == ord('v'):
            self.switch_cameras()
        time.sleep(.005)
        return False

    # ---------------------------------------------------------------- tracking and control
    def track(self, observation, now):
        args = self.args
        age_ms = (time.perf_counter_ns()-observation.frame.received_ns)/1e6
        new_result = observation.frame.index != self.last_result
        fresh = age_ms <= self.max_age_ms
        arm = observation.arm if fresh else None
        hand = observation.hand if fresh else None
        reason = observation.reason if fresh else 'Tracking result too old for control (camera or computer too slow)'
        if arm is not None and self.anchor is not None:
            try:
                self.anchor.validate_measurement(arm)
            except ValueError as error:
                arm, reason = None, str(error)
        # Partial holds: a briefly missing arm or hand is bridged with its last good
        # measurement, so one flickering part does not freeze the other.
        hold_s = args.part_hold_ms/1000
        if arm is not None:
            self.last_arm = (arm, now)
        if hand is not None:
            self.last_hand = (hand, now)
        arm_use = arm if arm is not None else (self.last_arm[0] if self.last_arm and now-self.last_arm[1] <= hold_s else None)
        hand_use = hand if hand is not None else (self.last_hand[0] if self.last_hand and now-self.last_hand[1] <= hold_s else None)
        valid = arm_use is not None and hand_use is not None and (arm is not None or hand is not None)
        self.partial = ('' if not valid or (arm is not None and hand is not None) else
                        'Hand not seen this moment - fingers holding their last pose' if hand is None else
                        'Arm not seen this moment - arm holding, fingers still following')
        self.age_ms, self.valid = age_ms, valid
        if not self.gate.active:
            self.pose_checks = start_pose_checks(arm, hand)
        self.gate.observe(valid, now)
        completed = self.calibration.update(now, arm, hand, new_result, observation.frame.received_ns/1e9, reason)
        if completed:
            self.calibration_step(*completed, observation)
        elif self.calibration.phase == 'cancelled':
            self.note = self.calibration.message
        if new_result:
            self.last_result = observation.frame.index
            if arm is not None and hand is not None:
                self.times.append(now)
        while self.times and self.times[0] < now-2:
            self.times.popleft()
        if self.gate.active and valid and observation.frame.index != self.last_solved:
            self.last_solved = observation.frame.index
            try:
                desired, self.fit = self.retarget.solve(arm_use, hand_use)
                gap = float(np.max(np.abs(desired[:7]-self.generator.last_q[:7])))
                if self.aligning and gap < ALIGN_DONE_RAD:
                    self.aligning = False
                    self.note = 'Robot matches your pose. Teleoperating; Space or S stops.'
                speed = self.robot.config['align_speed_scale'] if self.aligning else 1.
                self.active_signal = self.generator.generate(desired, observation.frame.received_ns, speed_scale=speed)
                self.simulation.apply(self.active_signal)
                self.simulation.set_overlay(self.retarget.overlay)
                self.control_reason = ''
            except ValueError as error:
                self.control_reason = str(error)
                self.hold(self.control_reason)
        if self.gate.active and not valid:
            self.hold('Brief tracking gap - holding pose')
        if not self.gate.active:
            self.active_signal = None
            self.simulation.hold('Tracking lost - holding pose' if self.gate.fault else 'Target stream stopped')
            if self.anchor is None:
                self.simulation.set_overlay(None)
        state = self.gate.state(self.calibrated, self.calibration.active)
        if self.fault and not self.gate.active:
            state, reason = 'FAULT', f'Recovered from an internal error: {self.fault}. Space resumes; R resets.'
        elif state == 'STREAMING' and self.aligning:
            state, reason = 'ALIGNING', 'Robot is moving slowly to match your start pose - hold still'
        elif state == 'STREAMING' and (self.control_reason or (self.fit and self.fit.get('limit_active'))):
            state = 'LIMIT_ACTIVE'
            reason = self.control_reason or ('Constrained: '+'; '.join(self.fit['limit_reasons']))
        elif state == 'STREAMING':
            reason = self.partial or 'Following coordinated arm + LEAP control signal'
        elif state == 'TRACKING_HOLD':
            reason = 'Holding: '+reason
        self.state, self.reason = state, reason

    def calibration_step(self, stage, result, observation):
        if stage == 'open':
            self.anchor, self.opened = result['anchor'], result['pose']
            self.reference = self.retarget.calibrate(self.anchor, result['hand'])
            self.hand_calibration = None
            wrist = (16 if self.args.side == 'right' else 15)
            if observation.pose_image is not None and not observation.views:
                self.start_wrist = observation.pose_image[wrist, :2].tolist()
            self.simulation.set_overlay(self.retarget.reference_overlay())
            match = ('The robot will match this pose' if self.reference.get('quality') == 'good' else
                     f"The robot can only approximate this pose ({self.reference.get('wrist_error_mm', 0):.0f} mm, "
                     f"{self.reference.get('palm_error_deg', 0):.0f} deg)") if self.reference.get('matched') else 'Robot uses its home pose'
            self.note = f'Start pose saved. {match}. Keep the arm in view, make a fist (thumb bent) and press C.'
        else:
            try:
                self.hand_calibration = make_calibration(self.args.side.title(), self.opened, result['pose'])
                self.retarget.calibrate_closed(result['hand'])
                self.note = 'Calibration complete. Return to the start pose, then press Space: the robot first moves slowly to match you.'
            except ValueError as error:
                self.note = f'{error}. Press C to retry.'
                self.calibration.message, self.calibration.phase = self.note, 'error'

    def publish_state(self, now):
        fit = self.fit if self.gate.active else None
        extra = []
        if self.state in ('LIMIT_ACTIVE', 'ALIGNING') and fit and fit.get('guidance'):
            extra.append(("Do this", fit['guidance'][:80]))
        self.simulation.set_hud("QuadraDex follower - simulation only", hud_lines(self.state, fit, [("Tracking", self.reason[:70])]+extra))
        self.simulation.step(min(.05, now-self.last_tick))
        self.last_tick = now

    def stereo_ui(self):
        if self.mode != 2:
            return {'mode': 1}
        info = {'mode': 2, 'message': self.stereo_message, 'warning': self.stereo_warning,
                'progress': self.collector.progress if self.collector else (1. if self.stereo else 0.)}
        if self.stereo is not None:
            info.update(quality=self.stereo.quality, rms_px=self.stereo.rms_px, angle_deg=self.stereo.angle_deg,
                        live_px=(self.composite.fusion or {}).get('median_reprojection_px') if self.composite else None)
        return info

    def ui_state(self, now):
        fit = self.fit if self.gate.active else None
        observation = self.observation
        return {'state': self.state, 'reason': self.reason, 'note': self.note, 'side': self.args.side,
                'mirrored': self.mirrored[0], 'camera_name': self.choice.name if self.choice else 'Camera',
                'camera_names': [v['choice'].name for v in self.views],
                'progress': self.calibration.status(now), 'calibrated': self.calibrated,
                'opened': self.anchor is not None, 'valid': getattr(self, 'valid', False), 'age_ms': getattr(self, 'age_ms', 0.),
                'metrics': {'valid_hz': len(self.times)/min(2, max(.01, now-self.started)),
                            'pose_ms': observation.timing_ms.get('pose', 0) if observation else 0,
                            'hand_ms': observation.timing_ms.get('hand', 0) if observation else 0},
                'fit': fit,
                'travel': finger_travel(self.opened, observation.hand if observation else None) if self.anchor is not None and not self.calibrated else {},
                'pose_checks': self.pose_checks, 'start_wrist': self.start_wrist, 'reference': self.reference,
                'guidance': fit.get('guidance', '') if fit and self.state in ('LIMIT_ACTIVE', 'ALIGNING') else '',
                'near_limits': fit.get('near_limits', []) if fit else [],
                'warning': self.log_warning, 'partial': self.partial if self.gate.active else '',
                'stereo': self.stereo_ui()}

    def restart_workers(self):
        for k, view in enumerate(self.views):
            if view['worker'] is not None:
                view['worker'].close()
            view['worker'] = self.make_worker(k, view['camera'])
        self.composite = self.pair_key = None
        if self.fusion is not None:
            self.fusion = StereoFusion(self.stereo, self.args.side, USE_MIN_CONFIDENCE)

    def switch_cameras(self):
        self.close_devices()
        self.known_choices = []
        self.stereo = self.fusion = self.collector = None
        self.stereo_job = None
        try:
            self.open_views(pick=not self.args.headless)
            self.device_error = None
        except Exception as error:
            self.device_failed(str(error) or 'No camera selected', time.perf_counter())
        self.started = time.perf_counter()

    def handle_key(self, key):
        """Returns True when the operator asked to quit."""
        if key in (27, ord('q')):
            return True
        now = time.perf_counter()
        if key == ord('o'):
            self.gate.stop()
            self.hold('Calibrating')
            self.anchor = self.hand_calibration = self.opened = None
            self.fit, self.fault = None, None
            self.last_arm = self.last_hand = None
            self.simulation.set_overlay(None)
            self.calibration.start('open', now)
        elif key == ord('c'):
            if self.anchor is None:
                self.note = 'Capture the start pose + open hand first: press O.'
            else:
                self.gate.stop()
                self.hold('Calibrating')
                self.hand_calibration = None
                self.calibration.start('closed', now)
        elif key == ord(' ') and not self.calibration.active:
            if self.gate.active:
                self.gate.stop()
                self.hold('Stopped by operator')
                self.note = 'Stopped. Space resumes; O recalibrates.'
            elif self.gate.start(self.calibrated, self.valid, now):
                self.fault = None
                self.generator.reconcile()
                self.retarget.seed = np.r_[self.generator.last_q[:7], self.retarget.reference_q[7:]]
                self.last_solved = -1
                self.aligning = True
                self.note = 'Robot moving slowly to match your pose, then it follows you. Space or S stops.'
            else:
                self.note = ('Finish calibration first (O, then C).' if not self.calibrated else
                             'Need 0.5 s of continuous arm + finger tracking before starting.')
        elif key == ord('s'):
            self.gate.stop()
            self.calibration.cancel('Stopped. O recalibrates; Space resumes when ready.')
            self.hold('Stopped by operator')
        elif key == ord('b'):
            self.simulation.request_object_reset()
        elif key == ord('k'):
            if self.mode == 2 and self.views:
                self.hold('Two-camera setup - targets inhibited')
                self.simulation.set_overlay(None)
                self.start_stereo_setup()
            else:
                self.note = 'K (two-camera setup) applies only in two-camera mode. Restart and choose two cameras.'
        elif key in (ord('r'), ord('h'), ord('m'), ord('n'), ord('v')):
            self.hold('Reconfiguring - targets inhibited')
            self.simulation.set_overlay(None)
            self.reset_calibration()
            self.fault = None
            if key in (ord('h'), ord('m'), ord('n'), ord('v')):
                self.emit('STOPPED', 'Camera/side changed')
                if key == ord('h'):
                    self.args.side = 'left' if self.args.side == 'right' else 'right'
                if key in (ord('m'), ord('n')):
                    k = 0 if key == ord('m') else 1
                    if k < self.mode:
                        self.mirrored[k] = not self.mirrored[k]
                        if self.mode == 2:
                            self.stereo = self.fusion = None   # the saved geometry was for the other image orientation
                            self.collector = self.stereo_job = None
                if key == ord('v'):
                    self.switch_cameras()
                else:
                    try:
                        self.restart_workers()
                    except Exception as error:
                        self.device_failed(str(error), now)
                self.session, self.sequence = str(uuid.uuid4()), 0
            self.note = f'{self.args.side.title()} arm selected. ' + START_NOTE.split(': ', 1)[-1]
        return False

    def publish(self, now):
        # Recheck immediately before publication; the viewer/IK may have consumed time.
        if not self.gate.active or self.calibration.active:
            self.active_signal = None
        if self.active_signal is not None and time.perf_counter_ns() > self.active_signal['expires_monotonic_ns']:
            self.hold('Source expired before publication')
        if time.perf_counter()-self.last_publish < 1/self.args.output_hz:
            return
        observation = self.observation
        self.emit(self.state, self.reason,
                  calibration_progress=self.calibration.status(time.perf_counter()),
                  camera_mode=self.mode,
                  camera={'name': self.choice.name, 'index': self.choice.index, 'backend': self.choice.backend,
                          'mirrored_input_corrected': self.mirrored[0]},
                  cameras=[{'name': v['choice'].name, 'index': v['choice'].index, 'mirrored_input_corrected': self.mirrored[k]}
                           for k, v in enumerate(self.views)],
                  stereo=None if self.mode != 2 else {'calibration': {'quality': self.stereo.quality, 'rms_px': self.stereo.rms_px,
                                                                      'angle_deg': self.stereo.angle_deg} if self.stereo else None,
                                                      'fusion': observation.fusion, 'warning': self.stereo_warning},
                  tracking={'parts': [{'label': p[0], 'ok': p[1], 'score': p[2], 'source': p[3]} for p in observation.parts],
                            'arm_reason': observation.arm_reason, 'hand_reason': observation.hand_reason,
                            'partial_hold': self.partial, 'pose_model': self.args.pose_model, 'timing_ms': observation.timing_ms},
                  start_pose={'checks': [{'label': c[0], 'ok': c[1], 'detail': c[2]} for c in self.pose_checks],
                              'robot_reference': self.reference, 'aligning': self.aligning},
                  fault=self.fault)
        self.last_publish = time.perf_counter()


def run_live(args, robot, simulation, output):
    LiveSession(args, robot, simulation, output).run()


def main(argv=None):
    args = parse_args(argv)
    if args.list_cameras:
        for device in discover_cameras(args.backend):
            print(f'{device.index}: {device.name} ({device.backend})')
        return 0
    if args.download_models:
        print(ensure_models(args.pose_model))
        return 0
    robot = RobotModel(args.arm_config)
    output = PacketOutput(args.udp, args.log)
    simulation = None
    try:
        simulation = Simulation(robot, args.headless)
        (run_demo if args.demo else run_live)(args, robot, simulation, output)
        if args.snapshot:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            robot.snapshot(args.snapshot, simulation.overlay)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        # Only start-up problems (configuration, models, simulator window) reach this point.
        print(f'Error: {error}', file=sys.stderr, flush=True)
        return 1
    finally:
        if simulation is not None:
            simulation.close()
        output.close()


if __name__ == '__main__':
    raise SystemExit(main())
