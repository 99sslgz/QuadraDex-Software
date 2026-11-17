"""Offline robot-model, sensing, signal and validation tests (no physical actuators)."""
import copy
import unittest
import numpy as np

from cv_hand_tracking.hand_geometry import features
from teleoperation.arm_tracking import ArmAnchor, ArmContinuity, arm_features
from teleoperation.evaluate_tracking import evaluate
from teleoperation.robot_model import RobotModel
from teleoperation.retarget import Retargeter, SignalGenerator, leap_sim_to_sdk
from teleoperation.simulator import Simulation


def pose_fixture():
    world=np.zeros((33,3))
    world[11],world[12]=[.2,0,0],[-.2,0,0]
    world[23],world[24]=[.16,.5,0],[-.16,.5,0]
    world[14],world[16]=[-.28,.15,-.24],[-.22,.30,-.45]
    world[13],world[15]=[.28,.15,-.24],[.22,.30,-.45]
    image=np.full((33,3),.5)
    confidence=np.full(33,.99)
    return image,world,confidence


def hand_fixture():
    world=np.zeros((21,3))
    world[1:5]=[[.030,.015,0],[.052,.03,0],[.072,.04,0],[.09,.05,0]]
    for base,x,length in [(5,.032,.035),(9,.008,.040),(13,-.017,.036),(17,-.041,.028)]:
        world[base:base+4]=[[x,.07,0],[x,.07+length,0],[x,.09+length,0],[x,.107+length,0]]
    return features(world)


class ArmTrackingTests(unittest.TestCase):
    def test_body_frame_is_right_handed_and_rigid_transform_invariant(self):
        image,world,confidence=pose_fixture()
        first=arm_features(image,world,confidence)
        angle=.7
        rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        second=arm_features(image,world@rotation.T+[1,2,3],confidence)
        np.testing.assert_allclose(first['points_body_m'],second['points_body_m'],atol=1e-12)
        basis=np.asarray(first['basis_columns_in_model_camera'])
        np.testing.assert_allclose(basis.T@basis,np.eye(3),atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(basis),1.)
        self.assertEqual(first['accuracy_status'],'not_independently_validated')

    def test_obscured_out_of_frame_and_impossible_arm_are_rejected(self):
        image,world,confidence=pose_fixture()
        confidence[14]=.1
        with self.assertRaises(ValueError):arm_features(image,world,confidence)
        confidence[14]=.99;image[16,0]=1.1
        with self.assertRaises(ValueError):arm_features(image,world,confidence)
        image[16,0]=.5;world[16]=world[14]
        with self.assertRaises(ValueError):arm_features(image,world,confidence)

    def test_neutral_calibration_and_length_change_check(self):
        arm=arm_features(*pose_fixture())
        hand=hand_fixture()
        anchor=ArmAnchor.capture([arm]*12,[hand]*12)
        anchor.validate_measurement(arm)
        changed={**arm,'forearm_model_m':arm['forearm_model_m']*1.4}
        with self.assertRaises(ValueError):anchor.validate_measurement(changed)
        moved=copy.deepcopy(arm)
        moved['points_body_m'][2][0]+=.1
        with self.assertRaises(ValueError):ArmAnchor.capture([arm]*11+[moved],[hand]*12)

    def test_temporal_jump_is_rejected(self):
        arm=arm_features(*pose_fixture())
        continuity=ArmContinuity()
        continuity.check(arm,1.)
        moved=copy.deepcopy(arm);moved['points_body_m'][2][0]+=.4
        with self.assertRaises(ValueError):continuity.check(moved,1.02)


class RobotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.robot=RobotModel()

    def setUp(self):
        self.robot.reset()

    def test_official_model_has_16_named_leap_joints_and_independent_motor_order(self):
        robot=self.robot
        self.assertEqual(robot.model.nq,23)
        self.assertEqual(robot.model.nu,23)
        self.assertEqual(robot.hand_joint_names,[f'leap_{i}' for i in range(16)])
        self.assertGreater(robot.qadr[7],robot.qadr[8])  # URDF traversal swaps motor 0/1.
        self.assertAlmostEqual(robot.lower[7],-1.047)
        self.assertAlmostEqual(robot.upper[8],2.23)

    def test_arm_inverse_kinematics_reaches_a_known_forward_pose(self):
        robot=self.robot;solver=Retargeter(robot)
        known=robot.home.copy();known[:7]+=[.15,.08,.05,.12,.04,-.06,.08]
        data=robot.forward(known)
        position=data.site('tool').xpos.copy();rotation=data.site('tool').xmat.reshape(3,3).copy()
        solved,error,rotation_error=solver.solve_arm(position,rotation,iterations=60)
        self.assertLess(error,.001)
        self.assertLess(rotation_error,.01)
        self.assertTrue(np.all(solved>=robot.lower) and np.all(solved<=robot.upper))

    def test_hand_inverse_kinematics_matches_independent_finger_targets(self):
        robot=self.robot;solver=Retargeter(robot)
        known=robot.home.copy()
        known[7:11]=[.15,.5,.7,.4]
        known[11:15]=[-.1,.4,.6,.3]
        known[19:23]=[.3,.3,.4,.3]
        data=robot.forward(known);root=data.body('leap_palm_lower')
        rotation=root.xmat.reshape(3,3).copy();position=root.xpos.copy()
        tips={finger:rotation.T@(data.site('tip_'+finger).xpos-position) for finger in solver.open_tips}
        solved,errors=solver.solve_hand(tips,iterations=45)
        self.assertLess(max(errors.values()),.001)
        self.assertTrue(np.all(solved>=robot.lower) and np.all(solved<=robot.upper))

    def test_calibrated_neutral_geometry_maps_to_robot_home(self):
        robot=self.robot;solver=Retargeter(robot)
        arm=arm_features(*pose_fixture());hand=hand_fixture()
        anchor=ArmAnchor.capture([arm]*12,[hand]*12)
        solver.calibrate(anchor,hand)
        solved,metrics=solver.solve(arm,hand)
        np.testing.assert_allclose(solved,robot.home,atol=1e-7)
        self.assertLess(metrics['arm_ik_error_mm'],.001)
        self.assertLess(metrics['mean_fingertip_fit_error_mm'],.001)

    def test_workspace_violation_blocks_the_whole_pair(self):
        robot=self.robot;solver=Retargeter(robot)
        arm=arm_features(*pose_fixture());hand=hand_fixture()
        solver.calibrate(ArmAnchor.capture([arm]*12,[hand]*12),hand)
        moved=copy.deepcopy(arm);moved['points_body_m'][2][0]+=3
        with self.assertRaises(ValueError):solver.solve(moved,hand)

    def test_sdk_conversion_and_velocity_limit(self):
        robot=self.robot;generator=SignalGenerator(robot)
        desired=robot.home+.1
        packet=generator.generate(desired,1_000_000_000,1_005_000_000)
        q=np.asarray(packet['positions_rad'])
        self.assertLessEqual(np.max(np.abs(q[:7]-robot.home[:7])),.0200001)
        self.assertLessEqual(np.max(np.abs(q[7:]-robot.home[7:])),.0400001)
        np.testing.assert_allclose(packet['leap_v1_sdk_positions_rad'],q[7:]+np.pi)
        self.assertFalse(packet['hardware_enabled'])
        self.assertEqual(packet['destination'],'simulation_only')
        with self.assertRaises(ValueError):leap_sim_to_sdk([0]*15)

    def test_stale_invalid_or_out_of_limit_commands_are_rejected(self):
        robot=self.robot;generator=SignalGenerator(robot)
        with self.assertRaises(ValueError):generator.generate(robot.home,1_000_000_000,1_200_000_000)
        wrong=robot.home.copy();wrong[0]=np.nan
        with self.assertRaises(ValueError):generator.generate(wrong,1_000_000_000,1_001_000_000)
        wrong=robot.home.copy();wrong[0]=20
        with self.assertRaises(ValueError):generator.generate(wrong,1_000_000_000,1_001_000_000)

    def test_simulator_applies_exported_signal_and_holds_on_expiry(self):
        robot=self.robot;generator=SignalGenerator(robot);sim=Simulation(robot,True)
        signal=generator.generate(robot.home+.1,1_000_000_000,1_001_000_000)
        sim.apply(signal,1_002_000_000)
        sim.step(.04,1_003_000_000)
        np.testing.assert_allclose(robot.data.ctrl[robot.actuators],signal['positions_rad'])
        self.assertGreater(np.linalg.norm(robot.measured()-robot.home),0)
        sim.step(.02,1_200_000_000)
        self.assertIsNone(sim.expires_ns)
        with self.assertRaises(ValueError):sim.apply(signal,1_200_000_000)
        bad={**signal,'joint_names':list(reversed(signal['joint_names']))}
        with self.assertRaises(ValueError):sim.apply(bad,1_002_000_000)


class AccuracyEvaluationTests(unittest.TestCase):
    def rows(self,error=.01):
        return [{'point':point,'estimated_timestamp_ms':i*40,'reference_timestamp_ms':i*40+2,
                 'estimated_x_m':error,'estimated_y_m':0,'estimated_z_m':0,
                 'reference_x_m':0,'reference_y_m':0,'reference_z_m':0}
                for point in ('shoulder','elbow','wrist') for i in range(10)]

    def test_report_checks_actual_error_and_never_enables_hardware(self):
        passed=evaluate(self.rows())
        self.assertTrue(passed['passed'])
        self.assertAlmostEqual(passed['statistics']['wrist']['max_mm'],10)
        self.assertFalse(passed['physical_actuation_enabled'])
        self.assertFalse(evaluate(self.rows(.08))['passed'])

    def test_missing_points_bad_values_and_skew_are_rejected(self):
        with self.assertRaises(ValueError):evaluate(self.rows()[:10])
        rows=self.rows();rows[0]['estimated_x_m']=float('nan')
        with self.assertRaises(ValueError):evaluate(rows)
        rows=self.rows();rows[0]['reference_timestamp_ms']=100
        with self.assertRaises(ValueError):evaluate(rows)


if __name__=='__main__':
    unittest.main()
