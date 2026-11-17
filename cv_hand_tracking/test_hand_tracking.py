"""Meaningful offline checks: geometry invariance, calibration, and stale gating.

Run: .venv\Scripts\python.exe -m unittest -v
Synthetic geometry validates the math, not webcam accuracy or PR-04 compliance.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

from hand_geometry import (Calibration, CONTROLLED, FINGERS, OneEuro, StreamGate,
                           bend, calibration_pose, features, make_calibration,
                           palm_frame, validate_geometry)
from receive_targets import usable
from calibration_ui import CalibrationCapture
from camera_picker import CameraChoice, choose_camera, discover_cameras
from track_hand import select_hand, parse_args


def open_hand():
    points = np.zeros((21, 3), dtype=float)
    points[1:5] = [[.030, .015, 0], [.052, .030, 0], [.072, .040, 0], [.090, .050, 0]]
    for base, x, length in [(5, .032, .035), (9, .008, .040), (13, -.017, .036), (17, -.041, .028)]:
        points[base:base+4] = [[x, .070, 0], [x, .070+length, 0],
                              [x, .090+length, 0], [x, .107+length, 0]]
    return points


class GeometryTests(unittest.TestCase):
    def test_bend_known_angles(self):
        self.assertAlmostEqual(bend(np.array([0., 0, 0]), np.array([1., 0, 0]), np.array([2., 0, 0])), 0)
        self.assertAlmostEqual(bend(np.array([0., 0, 0]), np.array([1., 0, 0]), np.array([1., 1, 0])), 90)

    def test_palm_frame_and_local_coordinates_rigid_transform(self):
        world = open_hand()
        angle = .71
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
        first = features(world)
        transformed = features(world @ rotation.T + [1., -2., .4])
        np.testing.assert_allclose(first["landmarks_palm_model_m"], transformed["landmarks_palm_model_m"], atol=1e-12)
        basis, width = palm_frame(world)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(basis), 1.)
        self.assertGreater(width, 0)

    def test_individual_finger_bend_and_pinch(self):
        world = open_hand()
        first = features(world)
        world[7] = world[6] + [0, 0, -.02]
        world[8] = world[7] + [0, -.017, 0]
        moved = features(world)
        self.assertAlmostEqual(moved["joints_deg"]["index"]["pip"], 90)
        self.assertGreater(moved["curl_raw"]["index"], first["curl_raw"]["index"])
        self.assertAlmostEqual(moved["curl_raw"]["middle"], first["curl_raw"]["middle"])
        world[4] = world[8]
        self.assertAlmostEqual(features(world)["pinch_distance_palm_widths"]["index"], 0)

    def test_invalid_and_degenerate_geometry(self):
        image = np.full((21, 3), .5)
        world = open_hand()
        self.assertIsNone(validate_geometry(image, world))
        image[0, 0] = 1.01
        self.assertIsNotNone(validate_geometry(image, world))
        image[0, 0] = .5
        world[7] = world[6]
        self.assertIsNotNone(validate_geometry(image, world))
        world[0, 0] = np.nan
        self.assertIsNotNone(validate_geometry(image, world))

    def test_filter_attenuates_jitter_and_follows_motion(self):
        smooth = OneEuro(2., 12.)
        base = open_hand()
        outputs, inputs = [], []
        for i in range(120):
            sample = base + (0.001 if i % 2 else -0.001)
            outputs.append(smooth.update(sample, i / 30.)[8, 0])
            inputs.append(sample[8, 0])
        self.assertLess(np.std(outputs[30:]), np.std(inputs[30:]) * .4)
        moved = smooth.update(base + .05, 120 / 30.)
        self.assertGreater(moved[8, 0] - outputs[-1], .03)
        with self.assertRaises(ValueError):
            smooth.update(base, 120 / 30.)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.opened = {"curl": dict.fromkeys(FINGERS, .1), "width": .08}
        self.closed = {"curl": dict.fromkeys(FINGERS, .7), "width": .08}

    def test_profile_roundtrip_hand_and_target_scale(self):
        profile = make_calibration("Right", self.opened, self.closed)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile.save(path)
            loaded = Calibration.load(path, "Right")
            with self.assertRaises(ValueError):
                Calibration.load(path, "Left")
        data = features(open_hand())
        data["curl_raw"] = dict.fromkeys(FINGERS, .4)
        target = loaded.targets(data)
        self.assertEqual(list(target["fingertips"]), list(CONTROLLED))
        self.assertNotIn("pinky", target["fingertips"])
        self.assertAlmostEqual(target["curl_01"]["index"], .5)
        np.testing.assert_allclose(target["fingertips"]["index"], np.asarray(data["fingertips_palm_model_m"]["index"]) / .08)

    def test_calibration_rejects_insufficient_motion_and_unstable_samples(self):
        with self.assertRaises(ValueError):
            make_calibration("Right", self.opened, self.opened)
        sample = features(open_hand())
        with self.assertRaises(ValueError):
            calibration_pose([sample] * 7)
        samples = []
        for i in range(12):
            changed = {**sample, "curl_raw": dict.fromkeys(FINGERS, float(i % 2))}
            samples.append(changed)
        with self.assertRaises(ValueError):
            calibration_pose(samples)
        self.assertAlmostEqual(calibration_pose([sample] * 12)["width"], sample["palm_width_model_m"])


class PermissionTests(unittest.TestCase):
    def test_tracking_recovery_requires_deliberate_restart(self):
        gate = StreamGate()
        gate.observe(True, 0.)
        self.assertFalse(gate.start(True, True, .1))
        self.assertFalse(gate.start(False, True, 1.))
        self.assertTrue(gate.start(True, True, 1.))
        gate.observe(False, 1.1)
        self.assertFalse(gate.active)
        self.assertEqual(gate.state(True), "TRACKING_LOST")
        gate.observe(True, 1.2)
        gate.observe(True, 2.)
        self.assertFalse(gate.active)
        self.assertTrue(gate.start(True, True, 2.))
        gate.stop()
        self.assertFalse(gate.active)

    def test_receiver_rejects_stale_future_and_inhibited_packets(self):
        packet = {"schema": "quadradex.hand.v1", "state": "STREAMING", "tracking_valid": True,
                  "calibrated": True, "teleop_targets": {}, "max_age_ms": 100,
                  "source_received_monotonic_ns": 1_000_000_000, "published_monotonic_ns": 1_030_000_000}
        self.assertTrue(usable(packet, 1_050_000_000, 100))
        self.assertFalse(usable(packet, 1_150_000_000, 100))
        self.assertFalse(usable(packet, 900_000_000, 100))
        self.assertFalse(usable({**packet, "teleop_targets": None}, 1_050_000_000, 100))
        self.assertFalse(usable({**packet, "state": "TRACKING_LOST"}, 1_050_000_000, 100))
        self.assertFalse(usable({}, 1_050_000_000, 100))


class GuidedCalibrationTests(unittest.TestCase):
    def test_progress_pauses_through_tracking_gap_and_completes(self):
        capture = CalibrationCapture(duration_s=.5, prepare_s=.1)
        sample = features(open_hand())
        capture.start("open", 0.)
        self.assertIsNone(capture.update(.05, sample, True, .05, ""))
        self.assertEqual(capture.phase, "prepare")
        for i in range(6):
            now = .11 + i*.04
            self.assertIsNone(capture.update(now, sample, True, now, ""))
        before = capture.status(.4)["progress"]
        self.assertGreater(before, 0)
        capture.update(.8, None, True, .8, "No hand")
        self.assertEqual(capture.phase, "paused")
        self.assertEqual(capture.status(.8)["progress"], before)
        self.assertTrue(capture.active)
        completed = None
        for i in range(20):
            now = .9 + i*.04
            completed = capture.update(now, sample, True, now, "")
            if completed:
                break
        self.assertIsNotNone(completed)
        self.assertEqual(completed[0], "open")
        self.assertEqual(capture.phase, "complete")
        self.assertFalse(capture.active)

    def test_duplicates_do_not_advance_progress_and_timeout_is_visible(self):
        capture = CalibrationCapture(prepare_s=0., timeout_s=2.)
        sample = features(open_hand())
        capture.start("closed", 0.)
        capture.update(.1, sample, True, .1, "")
        for _ in range(20):
            capture.update(.12, sample, True, .1, "")
        self.assertEqual(len(capture.samples), 1)
        self.assertEqual(capture.valid_s, 0)
        capture.update(2.1, sample, True, 2.1, "")
        self.assertFalse(capture.active)
        self.assertIn("timed out", capture.message)

    def test_unstable_controlled_finger_restarts_progress(self):
        capture = CalibrationCapture(prepare_s=0.)
        capture.start("closed", 0.)
        sample = features(open_hand())
        for i in range(8):
            changed = {**sample, "curl_raw": {**sample["curl_raw"], "index": float(i%2)}}
            capture.update(i*.04, changed, True, i*.04, "")
        self.assertEqual(capture.phase, "steady")
        self.assertEqual(capture.valid_s, 0)
        self.assertEqual(len(capture.samples), 1)

    def test_unused_pinky_jitter_does_not_block_calibration(self):
        sample = features(open_hand())
        samples = [{**sample, "curl_raw": {**sample["curl_raw"], "pinky": float(i%2)}} for i in range(12)]
        self.assertGreater(calibration_pose(samples)["width"], 0)


class CameraAndSelectionTests(unittest.TestCase):
    def test_picker_changes_device_and_transfers_selected_preview(self):
        readers = []
        class Reader:
            def __init__(self, index, *args):
                self.index, self.error, self.closed = index, None, False
                readers.append(self)
            def get(self):
                import time
                return SimpleNamespace(image=np.zeros((48,64,3),dtype=np.uint8), received_ns=time.perf_counter_ns())
            def close(self):
                self.closed = True
        count = [0]
        def keys(_):
            import time
            time.sleep(.002)
            count[0] += 1
            return ord("2") if count[0] == 3 else 13 if count[0] > 6 else -1
        args = SimpleNamespace(backend="msmf",width=640,height=480,camera_fps=30)
        devices = [CameraChoice(0,"msmf","Built in"),CameraChoice(2,"msmf","USB")]
        with patch("camera_picker.discover_cameras",return_value=devices), patch("camera_picker.cv2.namedWindow"), patch("camera_picker.cv2.setMouseCallback"), patch("camera_picker.cv2.imshow"), patch("camera_picker.cv2.waitKeyEx",side_effect=keys), patch("camera_picker.cv2.getWindowProperty",return_value=1), patch("camera_picker.cv2.destroyWindow"):
            choice, reader = choose_camera(Reader,args)
        self.assertEqual(choice,devices[1])
        self.assertEqual(reader.index,2)
        self.assertFalse(reader.closed)
        self.assertTrue(all(other.closed for other in readers if other is not reader))
        reader.close()

    def test_camera_names_and_indices_use_same_explicit_backend(self):
        device = SimpleNamespace(index=3, name="USB test camera", backend=1400)
        with patch("cv2_enumerate_cameras.enumerate_cameras", return_value=[device]) as discover:
            choices = discover_cameras("msmf")
        discover.assert_called_once_with(1400)
        self.assertEqual((choices[0].index, choices[0].backend, choices[0].name), (3, "msmf", "USB test camera"))

    def test_default_ui_chooses_camera_and_hand(self):
        args = parse_args([])
        self.assertIsNone(args.camera)
        self.assertEqual(args.hand, "auto")
        self.assertEqual(parse_args(["--camera", "2"]).camera, 2)

    def test_auto_hand_selection_and_wrong_hand_diagnostic(self):
        world = open_hand()
        image = np.full((21,3), .5)
        def points(array):
            return [SimpleNamespace(x=float(x),y=float(y),z=float(z)) for x,y,z in array]
        result = SimpleNamespace(handedness=[[SimpleNamespace(category_name="Left",score=.95)]],
                                 hand_landmarks=[points(image)], hand_world_landmarks=[points(world)])
        selected, error = select_hand(result, None, False, .6)
        self.assertIsNone(error)
        self.assertEqual(selected[3], "Left")
        selected, error = select_hand(result, "Right", False, .6)
        self.assertIsNone(selected)
        self.assertIn("Left", error)
        self.assertIn("Right", error)
        selected, error = select_hand(result, "Right", True, .6)
        self.assertIsNone(error)
        self.assertEqual(selected[3], "Right")


if __name__ == "__main__":
    unittest.main()
