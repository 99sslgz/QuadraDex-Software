"""Offline robot-model, sensing, signal and validation tests (no physical actuators)."""
import copy
import sys
import cv2
import time
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/"cv_hand_tracking"))
from hand_geometry import features
from teleoperation.arm_tracking import ArmAnchor, ArmContinuity, arm_features, part_status
from teleoperation.calibration import PairedCalibration, TeleopGate, start_pose_checks
from teleoperation.evaluate_tracking import evaluate
from teleoperation.robot_model import RobotModel
from teleoperation.retarget import Retargeter, SignalGenerator, leap_sim_to_sdk
from teleoperation.simulator import Simulation
from teleoperation.vision import HandWindow, Observation, check_hand, select_arm_hand
from teleoperation.stereo import StereoCalibration, StereoCollector, StereoFusion, calibrate, pair_observations
from teleoperation.synthetic_scene import SIZE, TwoCameraScene
import tempfile
from teleoperation import synthetic_hand as synth
from teleoperation import run as app
from camera_picker import CameraChoice
from track_hand import Frame


def pose_fixture():
    """MediaPipe world axes: x right (operator's left), y down, z away from the camera."""
    world = np.zeros((33, 3))
    world[11], world[12] = [.2, 0, 0], [-.2, 0, 0]
    world[23], world[24] = [.16, .5, 0], [-.16, .5, 0]
    world[14], world[16] = [-.28, .15, -.24], [-.22, .30, -.45]
    world[13], world[15] = [.28, .15, -.24], [.22, .30, -.45]
    image = np.full((33, 3), .5)
    confidence = np.full(33, .99)
    return image, world, confidence


def start_pose_world():
    """Recommended start pose: upper arm down, forearm up and forward (MediaPipe camera axes)."""
    world = np.zeros((33, 3))
    world[11], world[12] = [.2, 0, 0], [-.2, 0, 0]
    world[14] = [-.22, .28, -.05]
    world[16] = world[14]+.25*np.array([0, -.7, -.7])
    world[13] = [.22, .28, -.05]
    world[15] = world[13]+.25*np.array([0, -.7, -.7])
    return world


def start_pose_arm(world=None, side='right'):
    return arm_features(np.full((33, 3), .5), start_pose_world() if world is None else world, np.full(33, .99), side)


def human_hand(fingers=None, thumb=None, rotation=None, local=None):
    """Synthetic operator right hand, palm toward the camera, fingers up (optionally rotated)."""
    points = synth.hand_landmarks(fingers, thumb) if local is None else local
    palm = synth.PALM_TOWARD_CAMERA if rotation is None else rotation@synth.PALM_TOWARD_CAMERA
    return features(synth.to_camera(points, palm))


FIST = {f: (0, 80, 95, 55) for f in synth.MCP}


def hand_fixture():
    world = np.zeros((21, 3))
    world[1:5] = [[.030, .015, 0], [.052, .03, 0], [.072, .04, 0], [.09, .05, 0]]
    for base, x, length in [(5, .032, .035), (9, .008, .040), (13, -.017, .036), (17, -.041, .028)]:
        world[base:base+4] = [[x, .07, 0], [x, .07+length, 0], [x, .09+length, 0], [x, .107+length, 0]]
    return features(world)


class ArmTrackingTests(unittest.TestCase):
    def test_body_frame_is_right_handed_and_invariant_to_heading_and_position(self):
        image, world, confidence = pose_fixture()
        first = arm_features(image, world, confidence)
        angle = .7  # operator turned about the vertical (camera y) axis
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        second = arm_features(image, world@rotation.T+[1, 2, 3], confidence)
        np.testing.assert_allclose(first['points_body_m'], second['points_body_m'], atol=1e-12)
        basis = np.asarray(first['basis_columns_in_model_camera'])
        np.testing.assert_allclose(basis.T@basis, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(basis), 1.)
        self.assertEqual(first['accuracy_status'], 'not_independently_validated')

    def test_hips_hidden_by_a_desk_or_outside_the_image_are_not_required(self):
        image, world, confidence = pose_fixture()
        confidence[23:25] = 0.
        image[23:25, 1] = 1.4
        arm = arm_features(image, world, confidence, 'right')
        wrist = arm['points_body_m'][2]
        self.assertGreater(wrist[0], 0)  # in front of the shoulders
        self.assertLess(wrist[1], 0)     # on the operator's right
        self.assertLess(wrist[2], 0)     # below the shoulders
        self.assertGreater(arm_features(image, world, confidence, 'left')['points_body_m'][2][1], 0)
        self.assertTrue(all(ok for _, ok, _, _ in part_status(image, confidence, 'right')))

    def test_part_status_names_the_missing_landmark(self):
        image, world, confidence = pose_fixture()
        confidence[14] = .2
        status = {label: ok for label, ok, _, _ in part_status(image, confidence, 'right')}
        self.assertFalse(status['Elbow'])
        self.assertTrue(status['Wrist'])
        with self.assertRaisesRegex(ValueError, 'elbow'):
            arm_features(image, world, confidence, 'right')

    def test_live_packets_with_numpy_values_serialize_to_json(self):
        """Regression: np.bool_ in the per-landmark status crashed Output.emit after camera selection."""
        import json
        from teleoperation.run import json_ready
        image, world, confidence = pose_fixture()
        outside = image.astype(np.float32)
        outside[16, 1] = 1.3  # wrist below the image edge: the old code returned np.bool_(False) here
        parts = part_status(outside, confidence.astype(np.float32), 'right')
        self.assertFalse(dict((p[0], p[1]) for p in parts)['Wrist'])
        self.assertTrue(all(type(ok) is bool and type(score) is float for _, ok, score, _ in parts))
        json.dumps([p[1] for p in parts])
        packet = {'tracking': {'parts': [{'label': p[0], 'ok': p[1], 'score': p[2]} for p in parts]},
                  'numpy': {'flag': np.bool_(True), 'value': np.float32(.5), 'count': np.int64(3),
                            'array': np.zeros((2, 2)), 'tuple': (np.bool_(False), 1)},
                  'arm_observation': arm_features(image, world, confidence, 'right')}
        decoded = json.loads(json.dumps(json_ready(packet), allow_nan=False))
        self.assertIs(decoded['numpy']['flag'], True)
        self.assertEqual(decoded['numpy']['tuple'], [False, 1])
        self.assertEqual(decoded['numpy']['array'], [[0., 0.], [0., 0.]])

    def test_obscured_out_of_frame_and_impossible_arm_are_rejected(self):
        image, world, confidence = pose_fixture()
        confidence[14] = .1
        with self.assertRaises(ValueError): arm_features(image, world, confidence)
        confidence[14] = .99
        image[16, 0] = 1.1
        with self.assertRaises(ValueError): arm_features(image, world, confidence)
        image[16, 0] = .5
        world[16] = world[14]
        with self.assertRaises(ValueError): arm_features(image, world, confidence)

    def test_neutral_calibration_and_length_change_check(self):
        arm = arm_features(*pose_fixture())
        hand = hand_fixture()
        anchor = ArmAnchor.capture([arm]*12, [hand]*12)
        anchor.validate_measurement(arm)
        changed = {**arm, 'forearm_model_m': arm['forearm_model_m']*3, 'upper_arm_model_m': arm['upper_arm_model_m']*1.5}
        with self.assertRaises(ValueError): anchor.validate_measurement(changed)
        moved = copy.deepcopy(arm)
        moved['points_body_m'][2][0] += .1
        with self.assertRaises(ValueError): ArmAnchor.capture([arm]*8+[moved]*4, [hand]*12)
        ArmAnchor.capture([arm]*11+[moved], [hand]*12)  # one outlier does not fail calibration

    def test_temporal_jump_is_rejected_but_a_confirmed_relocation_is_accepted(self):
        arm = arm_features(*pose_fixture())
        continuity = ArmContinuity()
        continuity.check(arm, 1.)
        moved = copy.deepcopy(arm)
        moved['points_body_m'][2][0] += .4
        with self.assertRaises(ValueError): continuity.check(moved, 1.02)
        continuity.check(moved, 1.05)  # the next estimate agrees with the new location
        continuity.check(moved, 1.08)


class VisionHelperTests(unittest.TestCase):
    @staticmethod
    def hand_result(wrists, labels):
        point = lambda x, y: SimpleNamespace(x=x, y=y, z=0.)
        hands = [[point(x, y)]+[point(x+.01*i, y-.01*i) for i in range(1, 21)] for x, y in wrists]
        return SimpleNamespace(hand_landmarks=hands, hand_world_landmarks=[[point(0, 0)]*21 for _ in wrists],
                               handedness=[[SimpleNamespace(category_name=label)] for label in labels])

    def test_hand_is_associated_by_wrist_distance_not_by_label(self):
        size = np.array([1000., 500.])
        result = self.hand_result([(.2, .5), (.7, .5)], ['Right', 'Right'])
        image, world, label, reason = select_arm_hand(result, np.array([690., 250.]), None, size, 80)
        self.assertEqual(reason, '')
        self.assertAlmostEqual(image[0, 0], .7)
        image, world, label, reason = select_arm_hand(result, np.array([450., 250.]), None, size, 80)
        self.assertIsNone(world)
        self.assertIn('does not belong', reason)

    def test_hand_window_is_square_inside_the_image_and_follows_smoothly(self):
        window = HandWindow()
        x1, y1, x2, y2 = window.update((720, 1280, 3), np.array([1250., 700.]), np.array([1150., 600.]))
        self.assertEqual(x2-x1, y2-y1)
        self.assertTrue(0 <= x1 < x2 <= 1280 and 0 <= y1 < y2 <= 720)
        hand = np.array([[1000., 400.], [1060., 330.]])
        first = window.update((720, 1280, 3), None, None, hand)
        second = window.update((720, 1280, 3), None, None, hand+[8, 0])
        self.assertLess(abs(second[0]-first[0]), 8)


class CalibrationAndGateTests(unittest.TestCase):
    def test_open_needs_a_still_arm_and_closed_needs_only_the_hand(self):
        arm = arm_features(*pose_fixture())
        hand = hand_fixture()
        capture = PairedCalibration(duration_s=.3, prepare_s=.1)
        capture.start('open', 0.)
        result, restarted = None, False
        for i in range(40):
            t = .2+i*.04
            sample = copy.deepcopy(arm)
            if i == 5:
                sample['points_body_m'][2][0] += .2
            result = capture.update(t, sample, hand, True, t, 'ok')
            restarted |= capture.phase == 'steady'
            if result:
                break
        self.assertTrue(restarted)
        self.assertEqual(result[0], 'open')
        self.assertIn('anchor', result[1])
        capture.start('closed', 10.)
        closed = None
        for i in range(40):
            t = 10.2+i*.04
            closed = capture.update(t, None, hand, True, t, 'arm hidden')
            if closed:
                break
        self.assertEqual(closed[0], 'closed')

    def test_brief_gap_holds_and_long_gap_latches(self):
        gate = TeleopGate(recovery_s=.5, grace_s=.3)
        gate.observe(True, 0.)
        self.assertTrue(gate.start(True, True, .6))
        gate.observe(False, 1.)
        self.assertEqual(gate.state(True), 'TRACKING_HOLD')
        gate.observe(False, 1.2)
        gate.observe(True, 1.25)
        self.assertEqual(gate.state(True), 'STREAMING')
        gate.observe(False, 2.)
        gate.observe(False, 2.35)
        self.assertEqual(gate.state(True), 'TRACKING_LOST')
        self.assertFalse(gate.active)
        gate.observe(True, 2.5)
        self.assertFalse(gate.start(True, True, 2.7))
        self.assertTrue(gate.start(True, True, 3.05))


class RobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot = RobotModel()

    def setUp(self):
        self.robot.reset()

    def calibrated(self, robot=None):
        solver = Retargeter(robot or self.robot)
        arm, hand = start_pose_arm(), human_hand()
        solver.calibrate(ArmAnchor.capture([arm]*12, [hand]*12), hand)
        solver.calibrate_closed(human_hand(FIST, synth.FIST_THUMB))
        return solver, arm, hand

    def test_official_model_has_16_named_leap_joints_and_independent_motor_order(self):
        robot = self.robot
        self.assertEqual(len(robot.qadr), 23)
        self.assertEqual(robot.model.nu, 23)
        self.assertEqual(robot.hand_joint_names, [f'leap_{i}' for i in range(16)])
        self.assertGreater(robot.qadr[7], robot.qadr[8])  # URDF traversal swaps motor 0/1.
        self.assertAlmostEqual(robot.lower[7], -1.047)
        self.assertAlmostEqual(robot.upper[8], 2.23)

    def test_home_hand_is_palm_down_with_fingers_forward_above_the_bench(self):
        data = self.robot.forward(self.robot.home)
        palm = data.body('leap_palm_lower').xmat.reshape(3, 3)
        self.assertGreater(palm[0, 0], .95)   # fingers point forward
        self.assertGreater(palm[2, 2], .95)   # LEAP palmar side (-z) faces the bench
        self.assertGreater(palm[1, 1], .95)   # thumb side to the robot's left (right hand, palm down)
        lowest = min(data.site('tip_'+f).xpos[2] for f in ('index', 'middle', 'ring', 'thumb'))
        self.assertGreater(lowest, self.robot.config['bench_height_m']+.05)

    def test_arm_inverse_kinematics_reaches_a_known_forward_pose(self):
        robot = self.robot
        solver = Retargeter(robot)
        known = robot.home.copy()
        known[:7] += [.15, .08, .05, .12, .04, -.06, .08]
        data = robot.forward(known)
        position = data.site('tool').xpos.copy()
        rotation = data.site('tool').xmat.reshape(3, 3).copy()
        solved, error, rotation_error = solver.solve_arm(position, rotation, iterations=60)
        self.assertLess(error, .001)
        self.assertLess(rotation_error, .01)
        self.assertTrue(np.all(solved >= robot.lower) and np.all(solved <= robot.upper))

    def test_each_leap_finger_follows_the_same_human_finger(self):
        solver, arm, hand = self.calibrated()
        q, _ = solver.solve(arm, human_hand({'index': (0, 70, 80, 45)}))
        self.assertGreater(q[8], .9)                                   # index MCP flexes
        self.assertGreater(q[9], .9)                                   # index PIP flexes
        self.assertLess(np.max(np.abs(q[11:19])), .02)                 # middle and ring stay straight
        q, _ = solver.solve(arm, human_hand({'ring': (0, 0, 75, 40)}))  # PIP/DIP only
        self.assertLess(abs(q[16]), .02)
        self.assertGreater(q[17], .9)
        self.assertLess(np.max(np.abs(q[7:15])), .02)
        q, _ = solver.solve(arm, human_hand({'index': (15, 0, 0, 0)}))  # index spreads toward the thumb
        self.assertLess(q[7], -.15)                                    # LEAP: negative abduction = toward the thumb

    def test_fist_closes_every_leap_finger_and_open_hand_opens_them(self):
        solver, arm, hand = self.calibrated()
        q, _ = solver.solve(arm, human_hand(FIST, synth.FIST_THUMB))
        for block in (8, 12, 16):
            np.testing.assert_allclose(q[block:block+3], [1.35, 1.45, 1.0], atol=.05)
        data = self.robot.forward(q)
        root = data.body('leap_palm_lower')
        thumb = root.xmat.reshape(3, 3).T@(data.site('tip_thumb').xpos-root.xpos)
        self.assertLess(thumb[2], solver.open_tips['thumb'][2]-.03)    # thumb swings to the palmar side
        q, metrics = solver.solve(arm, hand)
        self.assertLess(np.max(np.abs(q[7:19])), .02)
        self.assertLess(metrics['thumb_fit_error_mm'], 15)

    def test_pinch_puts_the_leap_thumb_on_the_pinched_fingertip(self):
        solver, arm, _ = self.calibrated()
        for finger in ('index', 'middle', 'ring'):
            local, _ = synth.pinch_landmarks(finger)
            q, metrics = solver.solve(arm, human_hand(local=local))
            self.assertEqual(metrics['pinch']['finger'], finger)
            self.assertLess(metrics['pinch']['robot_gap_mm'], 5., finger)
            self.assertTrue(np.all(q >= self.robot.lower) and np.all(q <= self.robot.upper))

    def test_robot_reference_pose_matches_the_operator_start_pose(self):
        solver, arm, hand = self.calibrated()
        self.assertEqual(solver.reference_report['quality'], 'good')
        palm = self.robot.forward(solver.reference_q).body('leap_palm_lower').xmat.reshape(3, 3)
        self.assertGreater(palm[2, 0], .9)     # LEAP fingers point up, like the operator's
        self.assertGreater(-palm[0, 2], .9)    # LEAP palm faces forward, like the operator's palm toward the camera
        self.assertGreater(palm[1, 1], .9)     # thumb to the left (right hand)
        q, metrics = solver.solve(arm, hand)
        np.testing.assert_allclose(q[:7], solver.reference_q[:7], atol=2e-3)
        self.assertLess(metrics['arm_ik_error_mm'], .5)
        self.assertFalse(metrics['limit_active'])

    def test_palm_rotation_is_followed_in_the_same_direction(self):
        solver, arm, _ = self.calibrated()
        angle = .7
        tilt = np.array([[1, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]])  # fingers toward the camera
        q, metrics = solver.solve(arm, human_hand(rotation=tilt))
        palm = self.robot.forward(q).body('leap_palm_lower').xmat.reshape(3, 3)
        self.assertGreater(palm[0, 0], .5)     # LEAP fingers now lean forward
        self.assertLess(metrics['wrist_orientation_error_deg'], 2)

    def test_reference_falls_back_to_home_when_matching_is_disabled(self):
        robot = RobotModel()
        robot.config['match_start_pose'] = False
        solver, arm, hand = self.calibrated(robot)
        np.testing.assert_allclose(solver.reference_q, robot.home)
        q, metrics = solver.solve(arm, hand)
        np.testing.assert_allclose(q[:7], robot.home[:7], atol=2e-3)

    def test_start_pose_checks_report_what_to_adjust(self):
        checks = dict((label, ok) for label, ok, _ in start_pose_checks(start_pose_arm(), human_hand()))
        self.assertTrue(all(checks.values()), checks)
        straight = start_pose_world()
        straight[16] = straight[14]+[0, .25, 0]                       # forearm hanging down: elbow not bent
        checks = dict((label, ok) for label, ok, _ in start_pose_checks(start_pose_arm(straight), human_hand(rotation=np.diag([-1., 1, -1]))))
        self.assertFalse(checks['Elbow bent'])
        self.assertFalse(checks['Palm toward camera'])

    def test_arm_motion_is_followed_in_every_direction(self):
        solver, arm, hand = self.calibrated()
        for delta in ([.12, 0, 0], [-.15, 0, 0], [0, -.15, 0], [0, .08, 0], [0, 0, -.12], [0, 0, .08]):  # camera axes
            moved = start_pose_world()
            moved[16] += delta
            moved[14] += np.array(delta)*.4
            solved, metrics = solver.solve(start_pose_arm(moved), hand)
            self.assertLess(metrics['arm_ik_error_mm'], 2., delta)
            self.assertTrue(np.all(solved >= self.robot.lower) and np.all(solved <= self.robot.upper))

    def test_workspace_violation_is_constrained_and_reported(self):
        robot = self.robot
        solver, arm, hand = self.calibrated()
        moved = copy.deepcopy(arm)
        moved['points_body_m'][2][2] -= 3
        solved, metrics = solver.solve(moved, hand)
        self.assertTrue(metrics['limit_active'])
        self.assertIn('workspace boundary', metrics['limit_reasons'])
        self.assertIn('up', metrics['guidance'])  # tells the operator the way back
        self.assertTrue(np.all(solved >= robot.lower) and np.all(solved <= robot.upper))
        self.assertGreaterEqual(robot.forward(solved).site('tool').xpos[2], robot.config['workspace_lower_m'][2]-.03)

    def test_reject_policy_blocks_the_whole_pair(self):
        solver, arm, hand = self.calibrated()
        solver.policy = 'reject'
        world = start_pose_world()
        world[16] = world[14]+[0, .25, 0]   # forearm swung down: wrist below the table-clearance floor
        with self.assertRaises(ValueError): solver.solve(start_pose_arm(world), hand)

    def test_sdk_conversion_and_velocity_limit(self):
        robot = self.robot
        generator = SignalGenerator(robot)
        desired = np.clip(robot.home+.1, robot.lower, robot.upper)
        packet = generator.generate(desired, 1_000_000_000, 1_005_000_000)
        q = np.asarray(packet['positions_rad'])
        dt = .02  # first packet uses the nominal step
        self.assertLessEqual(np.max(np.abs(q[:7]-robot.home[:7])), robot.config['max_velocity_rad_s']*dt+1e-7)
        self.assertLessEqual(np.max(np.abs(q[7:]-robot.home[7:])), robot.config['hand_max_velocity_rad_s']*dt+1e-7)
        np.testing.assert_allclose(packet['leap_v1_sdk_positions_rad'], q[7:]+np.pi)
        self.assertFalse(packet['hardware_enabled'])
        self.assertEqual(packet['destination'], 'simulation_only')
        with self.assertRaises(ValueError): leap_sim_to_sdk([0]*15)

    def test_stale_invalid_or_out_of_limit_commands_are_rejected(self):
        robot = self.robot
        generator = SignalGenerator(robot)
        with self.assertRaises(ValueError): generator.generate(robot.home, 1_000_000_000, 1_200_000_000)
        wrong = robot.home.copy()
        wrong[0] = np.nan
        with self.assertRaises(ValueError): generator.generate(wrong, 1_000_000_000, 1_001_000_000)
        wrong = robot.home.copy()
        wrong[0] = 20
        with self.assertRaises(ValueError): generator.generate(wrong, 1_000_000_000, 1_001_000_000)
        nudged = robot.home.copy()
        nudged[8] = robot.lower[8]-.01  # fractional overshoot is clamped, not fatal
        q = np.asarray(generator.generate(nudged, 1_000_000_000, 1_001_000_000)['positions_rad'])
        self.assertTrue(np.all(q >= robot.lower) and np.all(q <= robot.upper))

    def test_holding_a_pose_past_a_joint_limit_never_raises(self):
        """Regression: holding the measured pose of a finger pressed past its limit ended the program."""
        robot = self.robot
        generator = SignalGenerator(robot)
        sim = Simulation(robot, True)
        sim.apply(generator.generate(robot.home, 1_000_000_000, 1_001_000_000), 1_002_000_000)
        robot.data.qpos[robot.qadr[8]] = robot.lower[8]-.02     # contact pushed index MCP past its stop
        robot.data.qpos[robot.qadr[3]] = robot.upper[3]+.01
        sim.hold('test')
        sim.step(.04)
        ctrl = robot.data.ctrl[robot.actuators]
        self.assertTrue(np.all(ctrl >= robot.lower-1e-12) and np.all(ctrl <= robot.upper+1e-12))
        robot.advance(np.full(23, np.nan), .02)  # malformed target: held, no exception
        generator.reconcile()
        self.assertTrue(np.all(generator.last_q >= robot.lower))

    def test_simulator_applies_exported_signal_and_holds_on_expiry(self):
        robot = self.robot
        generator = SignalGenerator(robot)
        sim = Simulation(robot, True)
        signal = generator.generate(np.clip(robot.home+.1, robot.lower, robot.upper), 1_000_000_000, 1_001_000_000)
        sim.apply(signal, 1_002_000_000)
        sim.step(.04, 1_003_000_000)
        np.testing.assert_allclose(robot.data.ctrl[robot.actuators], signal['positions_rad'])
        self.assertGreater(np.linalg.norm(robot.measured()-robot.home), 0)
        sim.step(.02, 1_200_000_000)
        self.assertIsNone(sim.expires_ns)
        with self.assertRaises(ValueError): sim.apply(signal, 1_200_000_000)
        bad = {**signal, 'joint_names': list(reversed(signal['joint_names']))}
        with self.assertRaises(ValueError): sim.apply(bad, 1_002_000_000)


class FakeCamera:
    error = None

    def __init__(self, *args):
        pass

    def close(self):
        pass


class FakeWorker:
    """Synthetic paired observations; `hand` and `world` can be changed by the test."""

    def __init__(self, *args):
        self.error = None
        self.index = 0
        self.world = start_pose_world()
        self.hand = human_hand()
        self.hide_hand = self.hide_arm = False

    def get(self):
        self.index += 1
        image, confidence = np.full((33, 3), .5), np.full(33, .99)
        arm = None if self.hide_arm else arm_features(image, self.world, confidence)
        return Observation(Frame(self.index, np.zeros((72, 128, 3), np.uint8), time.perf_counter_ns(), 0.), time.perf_counter_ns(),
                           arm, None if self.hide_hand else self.hand, None, image, 'synthetic', None,
                           part_status(image, confidence, 'right'), '', '', None, {'pose': 1., 'hand': 1.})

    def history(self):
        return [self.get()]

    def close(self):
        pass


class LiveSessionTests(unittest.TestCase):
    """The live loop must keep running through limits, errors and camera failures."""

    def setUp(self):
        for item in (patch.object(app, 'WholeArmVision', FakeWorker), patch.object(app, 'LatestCamera', FakeCamera)):
            item.start()
            self.addCleanup(item.stop)
        robot = RobotModel()
        args = app.parse_args(['--cameras', '1', '--camera', '0', '--headless', '--duration', '60'])
        self.session = app.LiveSession(args, robot, Simulation(robot, True), app.PacketOutput(None, None))
        self.session.attach([(CameraChoice(0, 'auto', 'Fake'), FakeCamera())])

    def run_for(self, seconds, until=None):
        end = time.perf_counter()+seconds
        states = []
        while time.perf_counter() < end:
            self.assertFalse(self.session.step())
            if not states or states[-1] != self.session.state:
                states.append(self.session.state)
            if until and self.session.state == until:
                break
            time.sleep(.005)
        return states

    def calibrate_and_start(self):
        session = self.session
        session.calibration = PairedCalibration(duration_s=.2, prepare_s=0.)
        session.calibration.start('open', time.perf_counter())
        self.run_for(3, until='CALIBRATION_REQUIRED')
        self.assertIsNotNone(session.anchor)
        session.worker.hand = human_hand(FIST, synth.FIST_THUMB)
        session.calibration.start('closed', time.perf_counter())
        self.run_for(3, until='READY')
        self.assertTrue(session.calibrated)
        session.worker.hand = human_hand()
        self.run_for(.6)
        session.handle_key(ord(' '))
        return self.run_for(4, until='STREAMING')

    def test_start_aligns_slowly_then_streams(self):
        states = self.calibrate_and_start()
        self.assertIn('ALIGNING', states)
        self.assertEqual(self.session.state, 'STREAMING')

    def test_limits_warn_and_guide_without_stopping(self):
        self.calibrate_and_start()
        session = self.session
        session.worker.world = start_pose_world()
        session.worker.world[16] = session.worker.world[14]+[0, .25, 0]   # forearm swung down: below the table floor
        self.run_for(1.5, until='LIMIT_ACTIVE')
        self.assertEqual(session.state, 'LIMIT_ACTIVE')
        self.assertIn('up', session.ui_state(time.perf_counter())['guidance'])
        self.assertIsNotNone(session.active_signal)       # the constrained command is still executed
        session.worker.world = start_pose_world()
        self.run_for(2, until='STREAMING')
        self.assertEqual(session.state, 'STREAMING')

    def test_internal_error_holds_and_the_session_continues(self):
        self.calibrate_and_start()
        session = self.session
        with patch.object(session.retarget, 'solve', side_effect=RuntimeError('injected failure')):
            self.run_for(.3)
        self.assertEqual(session.state, 'FAULT')
        self.assertFalse(session.gate.active)
        self.assertIsNone(session.active_signal)
        self.run_for(.7)
        session.handle_key(ord(' '))
        self.run_for(4, until='STREAMING')
        self.assertEqual(session.state, 'STREAMING')

    def test_a_briefly_missing_hand_or_arm_does_not_freeze_the_other(self):
        self.calibrate_and_start()
        session = self.session
        session.worker.hide_hand = True
        session.worker.world = start_pose_world()
        session.worker.world[16] += [.05, 0, 0]
        states = self.run_for(.4)
        self.assertEqual(states[-1], 'STREAMING')                 # no TRACKING_HOLD / LOST for a missing hand
        self.assertIn('fingers holding', session.reason)
        self.assertIsNotNone(session.active_signal)              # the arm keeps following
        session.worker.hide_hand, session.worker.hide_arm = False, True
        self.run_for(.4)
        self.assertEqual(session.state, 'STREAMING')
        self.assertIn('Arm not seen', session.reason)
        self.run_for(.6)                                         # beyond the part hold: a real gap
        self.assertIn(session.state, ('TRACKING_HOLD', 'TRACKING_LOST'))
        session.worker.hide_arm = False

    def test_camera_failure_is_retried_not_fatal(self):
        session = self.session
        session.worker.error = 'Camera stopped returning frames'
        self.run_for(.2)
        self.assertEqual(session.state, 'CAMERA_ERROR')
        session.device_retry_at = 0.
        self.run_for(.5)
        self.assertIsNone(session.device_error)
        self.assertNotEqual(session.state, 'CAMERA_ERROR')


class LeftArmTests(unittest.TestCase):
    """A left arm drives the right LEAP hand in the same direction as the operator moves."""

    SWAP = [12 if i == 11 else 11 if i == 12 else 14 if i == 13 else 13 if i == 14 else 16 if i == 15 else 15 if i == 16 else i
            for i in range(33)]

    @classmethod
    def setUpClass(cls):
        cls.robot = RobotModel()

    def left_arm(self, world):
        """Mirror image of a right-arm scene = the same motion made with the left arm."""
        return arm_features(np.full((33, 3), .5), (world*[-1, 1, 1])[self.SWAP], np.full(33, .99), 'left')

    def left_hand(self, fingers=None, thumb=None, rotation=None, local=None):
        points = synth.hand_landmarks(fingers, thumb) if local is None else local
        palm = synth.PALM_TOWARD_CAMERA if rotation is None else rotation@synth.PALM_TOWARD_CAMERA
        return features(synth.to_camera(points, palm)*[-1, 1, 1])

    def setUp(self):
        self.solver = Retargeter(self.robot)
        self.arm, self.hand = self.left_arm(start_pose_world()), self.left_hand()
        self.solver.calibrate(ArmAnchor.capture([self.arm]*10, [self.hand]*10), self.hand)
        self.solver.calibrate_closed(self.left_hand(FIST, synth.FIST_THUMB))

    def palm(self, q):
        return self.robot.forward(q).body('leap_palm_lower').xmat.reshape(3, 3)

    def test_start_pose_matches_and_palm_faces_the_same_way(self):
        self.assertEqual(self.solver.reference_report['quality'], 'good')
        q, metrics = self.solver.solve(self.arm, self.hand)
        palm = self.palm(q)
        self.assertGreater(palm[2, 0], .9)      # fingers up like the operator's
        self.assertGreater(-palm[0, 2], .9)     # palm faces forward like the operator's
        self.assertFalse(metrics['limit_active'])

    def test_motion_and_rotation_go_the_same_way_as_the_operator(self):
        world = start_pose_world()
        world[16] += [-.15, 0, 0]               # mirrored scene: the left hand moves to the operator's left
        world[14] += [-.06, 0, 0]
        start = self.robot.forward(self.solver.reference_q).site('tool').xpos.copy()
        q, metrics = self.solver.solve(self.left_arm(world), self.hand)
        moved = self.robot.forward(q).site('tool').xpos
        self.assertGreater(moved[1]-start[1], .1)   # robot moves to its left too
        self.assertFalse(metrics['limit_active'])
        self.solver.seed = self.solver.reference_q.copy()   # back at the start pose (streaming input is continuous)
        yaw = np.array([[np.cos(.6), 0, np.sin(.6)], [0, 1, 0], [-np.sin(.6), 0, np.cos(.6)]])
        hand = self.left_hand(rotation=yaw)
        q, metrics = Retargeter.solve(self.solver, self.arm, hand)
        palm_body = np.asarray(self.arm['basis_columns_in_model_camera']).T@np.asarray(hand['palm_basis_columns_in_model_camera'])
        operator_palmar = -palm_body[:, 2]      # left hand: hand_geometry normal is dorsal
        np.testing.assert_allclose(-self.palm(q)[:, 2], operator_palmar, atol=.08)  # palm body z is ~3 deg off the anatomical normal
        self.assertLess(metrics['wrist_orientation_error_deg'], 2)

    def test_left_fingers_and_pinch_map_like_the_right_hand(self):
        q, _ = self.solver.solve(self.arm, self.left_hand({'index': (0, 70, 80, 45)}))
        self.assertGreater(q[8], .9)
        self.assertLess(np.max(np.abs(q[11:19])), .02)
        q, metrics = self.solver.solve(self.arm, self.left_hand(local=synth.pinch_landmarks('index')[0]))
        self.assertLess(metrics['pinch']['robot_gap_mm'], 5)
        q, _ = self.solver.solve(self.arm, self.left_hand({'index': (14, 0, 0, 0)}))
        self.assertLess(q[7], -.15)             # spreads toward the thumb, like the right hand


class TrackingStabilityTests(unittest.TestCase):
    def test_hand_window_stays_still_while_the_hand_moves_inside_it(self):
        window = HandWindow()
        hand = np.array([[600., 300.], [660., 380.]])
        first = window.update((720, 1280, 3), None, None, hand)
        for dx in (6, 12, 18, 24):
            self.assertEqual(window.update((720, 1280, 3), None, None, hand+[dx, 0]), first)
        moved = window.update((720, 1280, 3), None, None, hand+[120, 0])   # near the edge: re-centre
        self.assertNotEqual(moved, first)

    def test_lost_hand_search_window_grows(self):
        window = HandWindow()
        sizes = [np.diff(window.update((720, 1280, 3), np.array([600., 400.]), np.array([600., 550.]))[::2])[0] for _ in range(4)]
        self.assertTrue(all(b > a for a, b in zip(sizes, sizes[1:])))

    def test_fingertips_slightly_past_the_image_edge_are_accepted(self):
        world = synth.to_camera(synth.hand_landmarks(), synth.PALM_TOWARD_CAMERA)-[0, 0, .5]
        image = np.column_stack((np.linspace(.6, 1.03, 21), np.full(21, .5), np.zeros(21)))
        self.assertIsNone(check_hand(image, world))
        image[:, 0] = np.linspace(.6, 1.2, 21)
        self.assertIn('outside', check_hand(image, world))

    def test_hand_near_the_previous_hand_is_kept_when_the_pose_wrist_is_off(self):
        size = np.array([1000., 500.])
        result = VisionHelperTests.hand_result([(.7, .5)], ['Right'])
        _, world, _, reason = select_arm_hand(result, np.array([450., 250.]), np.array([705., 250.]), size, 80)
        self.assertEqual(reason, '')
        self.assertIsNotNone(world)

    def test_segment_lengths_are_stabilized_to_the_calibration(self):
        robot = RobotModel()
        solver = Retargeter(robot)
        arm, hand = start_pose_arm(), human_hand()
        solver.calibrate(ArmAnchor.capture([arm]*12, [hand]*12), hand)
        noisy = copy.deepcopy(arm)
        points = np.asarray(noisy['points_body_m'])
        points[2] = points[1]+1.15*(points[2]-points[1])        # depth noise stretches the forearm by 15 %
        noisy['points_body_m'] = points.tolist()
        q, metrics = solver.solve(noisy, hand)
        np.testing.assert_allclose(q[:7], solver.reference_q[:7], atol=1e-3)   # same direction: no wrist jump


class StereoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = TwoCameraScene()
        collector = StereoCollector(min_interval_s=0)
        t = 0.
        while not collector.complete:
            o1, o2 = cls.scene.observe(t)
            collector.update(o1, o2, 0., t)
            t += .1
        cls.calibration = calibrate(collector.samples, ['front [0]', 'side [1]'], [SIZE, SIZE])

    def test_calibration_from_the_operator_recovers_the_camera_geometry(self):
        c = self.calibration
        rotation_error = np.degrees(np.linalg.norm(cv2.Rodrigues(c.r@self.scene.R2.T)[0]))
        self.assertLess(rotation_error, 1.)
        self.assertAlmostEqual(c.angle_deg, 60, delta=2)
        self.assertAlmostEqual(c.baseline_m/np.linalg.norm(self.scene.t2), 1, delta=.06)
        self.assertLess(c.rms_px, 3)
        self.assertEqual(c.quality, 'good')

    def test_calibration_is_saved_reloaded_and_matched_to_the_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'stereo.json'
            self.calibration.save(path)
            loaded = StereoCalibration.load(path)
        self.assertTrue(loaded.matches(['front [0]', 'side [1]'], [SIZE, SIZE]))
        self.assertFalse(loaded.matches(['front [0]', 'side [2]'], [SIZE, SIZE]))
        self.assertFalse(loaded.matches(['front [0]', 'side [1]'], [SIZE, (640, 480)]))

    def fused_errors(self, start, occluded=None):
        self.scene.occluded = occluded or {1: set(), 2: set()}
        fusion = StereoFusion(self.calibration, 'right')
        forearm, pip, sources = [], [], None
        for k in range(60):
            t = start+k*.05
            o1, o2 = self.scene.observe(t)
            result = fusion.fuse(o1, o2, t)
            X, hand = self.scene.body(t)
            if result.arm:
                vector = np.subtract(result.arm['wrist_model_m'], result.arm['elbow_model_m'])
                forearm.append(np.linalg.norm(vector-(X[16]-X[14])))
            if result.hand and k > 10:
                truth = features(hand)
                pip += [abs(result.hand['joints_deg'][f]['pip']-truth['joints_deg'][f]['pip']) for f in ('index', 'middle', 'ring')]
            sources = result.info
        self.scene.occluded = {1: set(), 2: set()}
        return np.median(forearm), np.median(pip), sources, len(forearm)

    def test_fused_arm_and_fingers_are_accurate_with_both_views(self):
        forearm, pip, info, n = self.fused_errors(40)
        self.assertEqual(n, 60)
        self.assertLess(forearm, .008)
        self.assertLess(pip, 5)
        self.assertEqual(set(info['pose_sources'].values()), {'both cameras'})
        self.assertEqual(info['hand']['source'], 'triangulated')

    def test_a_landmark_hidden_in_one_view_is_recovered_from_the_other(self):
        forearm, pip, info, n = self.fused_errors(50, {1: set(), 2: {14, 16, 'hand'}})   # elbow, wrist, hand hidden in camera 2
        self.assertEqual(n, 60)
        self.assertLess(forearm, .015)
        self.assertLess(pip, 5)
        self.assertEqual(info['pose_sources']['wrist'], 'camera 1 only')
        self.assertEqual(info['hand']['source'], 'camera 1')
        forearm, pip, info, n = self.fused_errors(60, {1: {16, 'hand'}, 2: set()})
        self.assertLess(forearm, .015)
        self.assertEqual(info['hand']['source'], 'camera 2')

    def test_left_arm_is_fused_too(self):
        fusion = StereoFusion(self.calibration, 'left')
        o1, o2 = self.scene.observe(70., side='left')
        result = fusion.fuse(o1, o2, 70.)
        self.assertIsNotNone(result.arm)
        self.assertEqual(result.arm['side'], 'Left')
        self.assertIsNotNone(result.hand)

    def test_observations_are_paired_by_capture_time(self):
        def obs(index, ms):
            return SimpleNamespace(frame=SimpleNamespace(index=index, received_ns=int(ms*1e6)))
        a = [obs(1, 0), obs(2, 33), obs(3, 66)]
        b = [obs(1, 10), obs(2, 45), obs(3, 80)]
        o1, o2, skew = pair_observations(a, b)
        self.assertEqual((o1.frame.index, o2.frame.index), (3, 3))
        self.assertAlmostEqual(skew, .014)
        o1, o2, skew = pair_observations(a, [obs(9, 400)])
        self.assertIsNone(o1)                   # nothing close in time: only the newer view is used
        self.assertEqual(o2.frame.index, 9)


class SceneDriver:
    """Feeds both fake two-camera workers from one synthetic scene; `t` and `curl` are set by the test."""

    def __init__(self):
        self.scene = TwoCameraScene()
        self.t, self.curl, self.side = 0., 0., 'right'
        self.cache = (None, None)
        self.history = ([], [])

    def pair(self):
        stamp = time.perf_counter()
        if self.cache[0] is None or stamp-self.cache[0] > .03:
            o1, o2 = self.scene.observe(self.t, side=self.side, curl=self.curl)
            self.cache = (stamp, (o1, o2))
            for k, o in enumerate((o1, o2)):
                self.history[k].append(o)
                del self.history[k][:-8]
        return self.cache[1]


class FakeStereoWorker:
    driver = None
    count = 0

    def __init__(self, *args):
        self.view = FakeStereoWorker.count % 2
        FakeStereoWorker.count += 1
        self.error = None

    def get(self):
        return self.driver.pair()[self.view]

    def history(self):
        self.driver.pair()
        return list(self.driver.history[self.view])

    def close(self):
        pass


class TwoCameraSessionTests(unittest.TestCase):
    def test_setup_calibrate_and_stream_through_an_occlusion(self):
        driver = SceneDriver()
        FakeStereoWorker.driver, FakeStereoWorker.count = driver, 0
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        for item in (patch.object(app, 'WholeArmVision', FakeStereoWorker), patch.object(app, 'LatestCamera', FakeCamera)):
            item.start()
            self.addCleanup(item.stop)
        robot = RobotModel()
        args = app.parse_args(['--cameras', '2', '--camera', '0', '--camera2', '1', '--headless', '--duration', '60',
                               '--stereo-file', str(Path(directory.name)/'stereo.json')])
        session = app.LiveSession(args, robot, Simulation(robot, True), app.PacketOutput(None, None))
        session.attach([(CameraChoice(0, 'auto', 'Front'), FakeCamera()), (CameraChoice(1, 'auto', 'Side'), FakeCamera())])
        session.collector = StereoCollector(min_interval_s=.0)
        session.stereo_message = ''
        end = time.perf_counter()+30
        while session.stereo is None and time.perf_counter() < end:   # operator moves during the setup
            driver.t += .1
            self.assertFalse(session.step())
            time.sleep(.002)
        self.assertIsNotNone(session.stereo)
        self.assertEqual(session.stereo.quality, 'good')
        self.assertTrue(Path(args.stereo_file).exists())

        def run(seconds, until=None):
            states, stop = [], time.perf_counter()+seconds
            while time.perf_counter() < stop:
                self.assertFalse(session.step())
                if not states or states[-1] != session.state:
                    states.append(session.state)
                if until and session.state == until:
                    break
                time.sleep(.005)
            return states
        driver.t = 100.
        session.calibration = PairedCalibration(duration_s=.3, prepare_s=0.)
        session.calibration.start('open', time.perf_counter())
        run(4, until='CALIBRATION_REQUIRED')
        self.assertIsNotNone(session.anchor)
        driver.curl = 1.
        session.calibration.start('closed', time.perf_counter())
        run(4, until='READY')
        self.assertTrue(session.calibrated)
        driver.curl = 0.
        run(.7)
        session.handle_key(ord(' '))
        states = run(5, until='STREAMING')
        self.assertEqual(session.state, 'STREAMING', states)
        driver.scene.occluded = {1: set(), 2: {14, 16, 'hand'}}     # arm folds: hidden in the side view
        run(.5)
        self.assertEqual(session.state, 'STREAMING')
        self.assertIsNotNone(session.active_signal)
        self.assertEqual(session.observation.fusion['pose_sources']['wrist'], 'camera 1 only')


class AccuracyEvaluationTests(unittest.TestCase):
    def rows(self, error=.01):
        return [{'point': point, 'estimated_timestamp_ms': i*40, 'reference_timestamp_ms': i*40+2,
                 'estimated_x_m': error, 'estimated_y_m': 0, 'estimated_z_m': 0,
                 'reference_x_m': 0, 'reference_y_m': 0, 'reference_z_m': 0}
                for point in ('shoulder', 'elbow', 'wrist') for i in range(10)]

    def test_report_checks_actual_error_and_never_enables_hardware(self):
        passed = evaluate(self.rows())
        self.assertTrue(passed['passed'])
        self.assertAlmostEqual(passed['statistics']['wrist']['max_mm'], 10)
        self.assertFalse(passed['physical_actuation_enabled'])
        self.assertFalse(evaluate(self.rows(.08))['passed'])

    def test_missing_points_bad_values_and_skew_are_rejected(self):
        with self.assertRaises(ValueError): evaluate(self.rows()[:10])
        rows = self.rows()
        rows[0]['estimated_x_m'] = float('nan')
        with self.assertRaises(ValueError): evaluate(rows)
        rows = self.rows()
        rows[0]['reference_timestamp_ms'] = 100
        with self.assertRaises(ValueError): evaluate(rows)


if __name__ == '__main__':
    unittest.main()
