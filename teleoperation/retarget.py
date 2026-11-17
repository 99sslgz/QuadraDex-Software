"""Bounded geometric IK and explicit LEAP V1 signal conventions."""
import time
import numpy as np
import mujoco

from teleoperation.arm_tracking import unit


def rotation_vector(rotation):
    quaternion=np.empty(4)
    mujoco.mju_mat2Quat(quaternion,np.asarray(rotation,float).ravel())
    if quaternion[0]<0:
        quaternion=-quaternion
    angle=2*np.arctan2(np.linalg.norm(quaternion[1:]),quaternion[0])
    return quaternion[1:]*(angle/max(1e-9,np.linalg.norm(quaternion[1:])))


def leap_sim_to_sdk(q):
    """Official V1 convention: logical simulation radians + pi, in motor ID order."""
    q=np.asarray(q,float)
    if q.shape!=(16,) or not np.isfinite(q).all():
        raise ValueError("LEAP V1 requires 16 finite positions in motor order 0..15")
    return q+np.pi


class Retargeter:
    def __init__(self,robot):
        self.robot=robot
        data=robot.forward(robot.home)
        self.home_tool=data.site('tool').xpos.copy()
        self.home_elbow=data.site('elbow').xpos.copy()
        self.home_rotation=data.site('tool').xmat.reshape(3,3).copy()
        root=data.body('leap_palm_lower')
        root_rotation=root.xmat.reshape(3,3)
        self.open_tips={finger:root_rotation.T@(data.site('tip_'+finger).xpos-root.xpos) for finger in ('index','middle','ring','thumb')}
        radial=unit(root_rotation.T@(data.body('leap_mcp_joint').xpos-data.body('leap_mcp_joint_3').xpos))
        distal=unit(self.open_tips['middle']-root_rotation.T@(data.body('leap_mcp_joint_2').xpos-root.xpos))
        distal=unit(distal-np.dot(distal,radial)*radial)
        self.human_to_model=np.column_stack((radial,distal,unit(np.cross(radial,distal))))
        self.robot_width=np.linalg.norm(data.body('leap_mcp_joint').xpos-data.body('leap_mcp_joint_3').xpos)
        self.seed=robot.home.copy()
        self.anchor=None
        self.human_open=None
        self.open_directions={}
        self.local_directions={}
        for finger in self.open_tips:
            site_id=robot.model.site('tip_'+finger).id
            offset=robot.model.site_pos[site_id]
            self.local_directions[finger]=unit(offset)
            body=robot.model.site_bodyid[site_id]
            self.open_directions[finger]=root_rotation.T@(data.xmat[body].reshape(3,3)@self.local_directions[finger])

    def calibrate(self,anchor,human_open):
        self.anchor,self.human_open=anchor,human_open
        self.seed=self.robot.home.copy()

    def desired_geometry(self,arm,hand):
        if self.anchor is None or self.human_open is None:
            raise ValueError("Open/neutral arm calibration is required")
        self.anchor.validate_measurement(arm)
        target=self.home_tool+self.robot.config['translation_scale']*(np.asarray(arm['points_body_m'][2])-self.anchor.wrist_body_m)
        if np.any(target<self.robot.config['workspace_lower_m']) or np.any(target>self.robot.config['workspace_upper_m']):
            raise ValueError("Arm target is outside the configured robot workspace")
        palm_body=np.asarray(arm['basis_columns_in_model_camera']).T@np.asarray(hand['palm_basis_columns_in_model_camera'])
        delta=palm_body@np.asarray(self.anchor.palm_in_body).T
        # Body axes and demo robot base axes share forward/left/up after calibration.
        orientation=delta@self.home_rotation
        scale=self.robot_width/self.human_open['palm_width_model_m']
        reflection=np.diag([1.,1.,-1. if self.anchor.side=='Left' else 1.])
        mapping=self.human_to_model@reflection
        tips,dirs={},{}
        for finger in self.open_tips:
            displacement=np.asarray(hand['fingertips_palm_model_m'][finger])-self.human_open['fingertips_palm_model_m'][finger]
            tips[finger]=self.open_tips[finger]+scale*(mapping@displacement)
            direction_delta=np.asarray(hand['distal_directions_palm'][finger])-self.human_open['distal_directions_palm'][finger]
            dirs[finger]=unit(self.open_directions[finger]+mapping@direction_delta)
        elbow=self.home_elbow+self.robot.config['translation_scale']*(np.asarray(arm['points_body_m'][1])-self.anchor.elbow_body_m)
        return target,orientation,tips,dirs,elbow

    def solve_arm(self,target,orientation,seed=None,iterations=32,elbow_target=None):
        robot=self.robot
        q=self.seed.copy() if seed is None else np.asarray(seed,float).copy()
        jp,jr=np.zeros((3,robot.model.nv)),np.zeros((3,robot.model.nv))
        for iteration in range(iterations):
            data=robot.forward(q)
            site=data.site('tool')
            pos_error=np.asarray(target)-site.xpos
            rot_error=rotation_vector(np.asarray(orientation)@site.xmat.reshape(3,3).T)
            if np.linalg.norm(pos_error)<.001 and np.linalg.norm(rot_error)<.015 and (elbow_target is None or iteration>20):
                break
            mujoco.mj_jacSite(robot.model,data,jp,jr,site.id)
            jac=np.vstack((jp[:,robot.dofs[:7]],.08*jr[:,robot.dofs[:7]]))
            error=np.r_[pos_error,.08*rot_error]
            step=jac.T@np.linalg.solve(jac@jac.T+.0004*np.eye(6),error)
            if elbow_target is not None:
                elbow=data.site('elbow')
                mujoco.mj_jacSite(robot.model,data,jp,jr,elbow.id)
                ej=jp[:,robot.dofs[:7]]
                null=np.eye(7)-np.linalg.pinv(jac,rcond=.02)@jac
                step+=.3*null@ej.T@np.linalg.solve(ej@ej.T+.002*np.eye(3),np.asarray(elbow_target)-elbow.xpos)
            q[:7]=np.clip(q[:7]+np.clip(step,-.18,.18),robot.lower[:7],robot.upper[:7])
        data=robot.forward(q)
        error=float(np.linalg.norm(data.site('tool').xpos-target))
        rotation_error=float(np.linalg.norm(rotation_vector(np.asarray(orientation)@data.site('tool').xmat.reshape(3,3).T)))
        return q,error,rotation_error

    def solve_hand(self,tips,directions=None,seed=None,iterations=20):
        robot=self.robot
        q=self.seed.copy() if seed is None else np.asarray(seed,float).copy()
        jp,jr=np.zeros((3,robot.model.nv)),np.zeros((3,robot.model.nv))
        for _ in range(iterations):
            data=robot.forward(q)
            root=data.body('leap_palm_lower')
            rotation=root.xmat.reshape(3,3)
            residuals,jacobians=[],[]
            for finger,target in tips.items():
                site=data.site('tip_'+finger)
                current=rotation.T@(site.xpos-root.xpos)
                residuals.append(np.asarray(target)-current)
                mujoco.mj_jacSite(robot.model,data,jp,jr,site.id)
                jacobians.append(rotation.T@jp[:,robot.dofs[7:]])
                if directions is not None:
                    body=robot.model.site_bodyid[site.id]
                    direction=data.xmat[body].reshape(3,3)@self.local_directions[finger]
                    current_direction=rotation.T@direction
                    residuals.append(.018*(np.asarray(directions[finger])-current_direction))
                    direction_jac=np.cross(jr[:,robot.dofs[7:]].T,direction).T
                    jacobians.append(.018*rotation.T@direction_jac)
            residual=np.concatenate(residuals)
            if np.linalg.norm(residual)<.001:
                break
            jac=np.vstack(jacobians)
            step=np.linalg.solve(jac.T@jac+.000025*np.eye(16),jac.T@residual)
            q[7:]=np.clip(q[7:]+np.clip(step,-.20,.20),robot.lower[7:],robot.upper[7:])
        data=robot.forward(q)
        root=data.body('leap_palm_lower')
        rotation=root.xmat.reshape(3,3)
        distances={finger:float(np.linalg.norm(rotation.T@(data.site('tip_'+finger).xpos-root.xpos)-target)) for finger,target in tips.items()}
        return q,distances

    def solve(self,arm,hand):
        target,orientation,tips,directions,elbow=self.desired_geometry(arm,hand)
        q,arm_error,rotation_error=self.solve_arm(target,orientation,elbow_target=elbow)
        q,tip_errors=self.solve_hand(tips,directions,q)
        metrics={'arm_ik_error_mm':arm_error*1000,'wrist_orientation_error_deg':float(np.degrees(rotation_error)),
                 'fingertip_fit_errors_mm':{k:v*1000 for k,v in tip_errors.items()},
                 'mean_fingertip_fit_error_mm':float(np.mean(list(tip_errors.values()))*1000)}
        if arm_error>self.robot.config['arm_fit_tolerance_m'] or rotation_error>.35:
            raise ValueError(f"Arm IK cannot reach this pose (position error {arm_error*1000:.0f} mm)")
        if np.mean(list(tip_errors.values()))>self.robot.config['hand_fit_tolerance_m']:
            raise ValueError(f"Finger target cannot be matched (mean fit {metrics['mean_fingertip_fit_error_mm']:.0f} mm)")
        self.seed=q.copy()
        return q,metrics


class SignalGenerator:
    def __init__(self,robot,max_age_ms=120):
        self.robot,self.max_age_ms=robot,max_age_ms
        self.last_q=robot.measured()
        self.last_time=None

    def reconcile(self):
        self.last_q=self.robot.measured()
        self.last_time=None

    def generate(self,desired,source_ns,now_ns=None):
        now_ns=time.perf_counter_ns() if now_ns is None else now_ns
        if not 0<=(now_ns-source_ns)/1e6<=self.max_age_ms:
            raise ValueError("Arm/hand source frame became stale before signal generation")
        desired=np.asarray(desired,float)
        if desired.shape!=(23,) or not np.isfinite(desired).all():
            raise ValueError("Invalid coordinated arm/hand target")
        if np.any(desired<self.robot.lower) or np.any(desired>self.robot.upper):
            raise ValueError("Joint target exceeds the robot model limits")
        dt=.02 if self.last_time is None else np.clip((now_ns-self.last_time)/1e9,0,.05)
        velocity=np.r_[np.full(7,self.robot.config['max_velocity_rad_s']),np.full(16,self.robot.config['hand_max_velocity_rad_s'])]
        limited=self.last_q+np.clip(desired-self.last_q,-velocity*dt,velocity*dt)
        self.last_q,self.last_time=limited.copy(),now_ns
        return {'schema':'quadradex.control.v1','destination':'simulation_only','hardware_enabled':False,
                'clock':'host_perf_counter_ns','source_received_monotonic_ns':int(source_ns),
                'generated_monotonic_ns':int(now_ns),'expires_monotonic_ns':int(source_ns+self.max_age_ms*1e6),
                'joint_names':self.robot.names,'positions_rad':limited.tolist(),
                'arm_positions_rad':limited[:7].tolist(),'leap_motor_ids':list(range(16)),
                'leap_sim_positions_rad':limited[7:].tolist(),'leap_v1_sdk_positions_rad':leap_sim_to_sdk(limited[7:]).tolist(),
                'leap_model_revision':'150bc3d4b61fd6619193ba5a8ef209f3609ced89',
                'velocity_limited':bool(np.any(np.abs(limited-desired)>1e-7))}
