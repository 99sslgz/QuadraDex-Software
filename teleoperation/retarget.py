"""Bounded geometric retargeting and explicit LEAP V1 signal conventions.

Arm: at calibration the robot computes a reference pose that matches the
operator's measured start posture (upper-arm direction, elbow bend, palm
orientation). While streaming, the wrist follows the operator's displacement
from the start pose (scaled by arm length) and the palm follows the operator's
rotation since the start pose, both applied to that reference.

Hand: index/middle/ring LEAP joints follow the operator's own joint angles,
each scaled by the open-hand and fist calibration so a full human fist gives a
full LEAP fist. The LEAP thumb has a different structure from a human thumb, so
it is solved by IK: its tip follows the operator's thumb tip (relative to the
index knuckle, scaled to LEAP size), and when the operator pinches, it is
pulled onto the LEAP fingertip being pinched.
"""
import time
import numpy as np
import mujoco

from teleoperation.arm_tracking import unit

ORIENTATION_WEIGHT = .20  # metres of position error equivalent to one radian
FINGER_BLOCKS = {"index": range(0, 4), "middle": range(4, 8), "ring": range(8, 12), "thumb": range(12, 16)}
LONG_FINGERS = ("index", "middle", "ring")
# LEAP flexion (rad) produced by a full calibrated human fist, per joint (MCP, PIP, DIP).
NOMINAL_FIST = (1.35, 1.45, 1.00)
DEFAULT_GAIN = 1.25          # before the fist calibration: MediaPipe tends to under-report flexion
THUMB_WEIGHT = 10.           # thumb-tip residual, per metre (1 cm ~ 0.1 rad of joint prior)
PINCH_START, PINCH_FULL = .60, .30   # human thumb-to-fingertip distance, in palm widths
MAX_SPREAD_RAD = .35
NEAR_LIMIT_RAD = .12
BODY_WORDS = (("forward", "back"), ("left", "right"), ("up", "down"))


def rotation_vector(rotation):
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(rotation, float).ravel())
    if quaternion[0] < 0:
        quaternion = -quaternion
    angle = 2*np.arctan2(np.linalg.norm(quaternion[1:]), quaternion[0])
    return quaternion[1:]*(angle/max(1e-9, np.linalg.norm(quaternion[1:])))


def rotate_like(vector, before, after):
    """Apply the smallest rotation that takes unit vector `before` to `after`."""
    axis = np.cross(before, after)
    sine, cosine = np.linalg.norm(axis), float(np.dot(before, after))
    if sine < 1e-8:
        return np.asarray(vector, float).copy()
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return (np.eye(3)+k+k@k*((1-cosine)/sine**2))@vector


def leap_sim_to_sdk(q):
    """Official V1 convention: logical simulation radians + pi, in motor ID order."""
    q = np.asarray(q, float)
    if q.shape != (16,) or not np.isfinite(q).all():
        raise ValueError("LEAP V1 requires 16 finite positions in motor order 0..15")
    return q+np.pi


def limited_dls(jac, error, q, lower, upper, damping=1e-4, max_step=.25):
    """Damped least squares step that drops joints already pushing into a limit."""
    free = np.ones(jac.shape[1], bool)
    step = np.zeros(jac.shape[1])
    for _ in range(3):
        j = jac[:, free]
        step[:] = 0
        step[free] = j.T@np.linalg.solve(j@j.T+damping*np.eye(j.shape[0]), error)
        blocked = free & (((q <= lower+1e-6) & (step < 0)) | ((q >= upper-1e-6) & (step > 0)))
        if not blocked.any():
            break
        free &= ~blocked
        if not free.any():
            step[:] = 0
            break
    return np.clip(step, -max_step, max_step), free


def palm_frame_for_robot(palm_in_body, side):
    """Target LEAP palm columns [radial, distal, palmar] in body axes.

    hand_geometry's palm normal is radial x distal: palmar for a right hand,
    dorsal for a left hand. For a left hand the right LEAP hand takes the same
    palm-facing and finger directions as the operator (a proper rotation); only
    the thumb side differs, because a right hand cannot copy a left hand's
    chirality. Rotations and translations of the left arm therefore go the same
    way on the robot as on the operator. (An earlier version mirrored the
    orientation but not the translation, so a left palm turning right made the
    robot palm turn left.)
    """
    palm = np.asarray(palm_in_body, float)
    if side == "Left":
        return palm@np.diag([-1., 1., -1.])
    return palm


def direction_words(vector, threshold=.04):
    """'forward and left' style description of a body-frame displacement."""
    words = []
    for axis in np.argsort(-np.abs(vector))[:2]:
        if abs(vector[axis]) >= threshold:
            words.append(BODY_WORDS[axis][0 if vector[axis] > 0 else 1])
    return " and ".join(words)


class Retargeter:
    def __init__(self, robot):
        self.robot = robot
        self.policy = robot.config.get("limit_policy", "constrain")
        self.match_start_pose = bool(robot.config.get("match_start_pose", True))
        data = robot.forward(robot.home)
        self.home_tool = data.site('tool').xpos.copy()
        self.home_elbow = data.site('elbow').xpos.copy()
        self.home_rotation = data.site('tool').xmat.reshape(3, 3).copy()
        self.shoulder = data.body('arm_base').xpos.copy()
        self.home_upper = unit(self.home_elbow-self.shoulder)
        self.home_forearm = unit(self.home_tool-self.home_elbow)
        self.robot_reach = robot.config["upper_arm_m"]+robot.config["forearm_m"]
        root = data.body('leap_palm_lower')
        root_rotation = root.xmat.reshape(3, 3)
        local = lambda point: root_rotation.T@(point-root.xpos)
        self.open_tips = {finger: local(data.site('tip_'+finger).xpos) for finger in ('index', 'middle', 'ring', 'thumb')}
        self.index_knuckle = local(data.body('leap_mcp_joint').xpos)
        radial = unit(local(data.body('leap_mcp_joint').xpos)-local(data.body('leap_mcp_joint_3').xpos))
        distal = unit(self.open_tips['middle']-local(data.body('leap_mcp_joint_2').xpos))
        distal = unit(distal-np.dot(distal, radial)*radial)
        # Columns: the human palm axes [radial, distal, palmar] expressed in LEAP palm coordinates.
        self.human_to_model = np.column_stack((radial, distal, unit(np.cross(radial, distal))))
        self.leap_index_length = float(np.linalg.norm(self.open_tips['index']-self.index_knuckle))
        # The LEAP thumb's base: its first two (CMC-like) joints sit together here.
        thumb_joint = robot.model.joint('leap_13').id
        self.thumb_base = local(data.xanchor[thumb_joint].copy())
        self.leap_thumb_reach = float(np.linalg.norm(self.open_tips['thumb']-self.thumb_base))
        self.human_thumb_length = .1
        # Fixed rotation between the IK tool site and the LEAP palm.
        self.tool_to_palm = self.home_rotation.T@root_rotation
        self.seed = robot.home.copy()
        self.anchor = None
        self.human_open = self.human_closed = None
        self.gains = {finger: [DEFAULT_GAIN]*3 for finger in LONG_FINGERS}
        self.thumb_scale = 1.
        self.scale = robot.config['translation_scale']
        self.overlay = None
        self.reference_q = robot.home.copy()
        self.reference_tool, self.reference_rotation = self.home_tool.copy(), self.home_rotation.copy()
        self.reference_report = {}
        self.anchor_wrist = np.zeros(3)

    # ------------------------------------------------------------------ calibration
    def calibrate(self, anchor, human_open):
        """OPEN step: arm anchor + open hand. Computes the matched robot reference pose."""
        self.anchor, self.human_open = anchor, human_open
        self.human_closed = None
        self.gains = {finger: [DEFAULT_GAIN]*3 for finger in LONG_FINGERS}
        self.scale = self.robot.config['translation_scale']
        if self.robot.config.get('scale_by_arm_length', True):
            self.scale *= float(np.clip(self.robot_reach/max(anchor.arm_length_m, 1e-3), .5, 2.))
        landmarks = np.asarray(human_open['landmarks_palm_model_m'])
        self.human_thumb_length = float(max(np.linalg.norm(landmarks[4]-landmarks[1]), .03))
        self.thumb_scale = self.leap_thumb_reach/self.human_thumb_length
        self.anchor_wrist = self.stabilized([anchor.shoulder_body_m, anchor.elbow_body_m, anchor.wrist_body_m])[2]
        self.reference_q, self.reference_report = self.match_reference(anchor)
        data = self.robot.forward(self.reference_q)
        self.reference_tool = data.site('tool').xpos.copy()
        self.reference_rotation = data.site('tool').xmat.reshape(3, 3).copy()
        self.seed = self.reference_q.copy()
        return self.reference_report

    def calibrate_closed(self, human_closed):
        """FIST step: per-joint gains so the operator's fist maps to a full LEAP fist."""
        self.human_closed = human_closed
        for finger in LONG_FINGERS:
            gains = []
            for joint, nominal in zip(("mcp", "pip", "dip"), NOMINAL_FIST):
                span = human_closed['joints_deg'][finger][joint]-self.human_open['joints_deg'][finger][joint]
                gains.append(float(np.clip(nominal/np.radians(span), .8, 2.2)) if span >= 15 else DEFAULT_GAIN)
            self.gains[finger] = gains
        return self.gains

    def posture_targets(self, shoulder, elbow, wrist):
        """Robot elbow / wrist positions with the operator's upper-arm and forearm directions.

        Both arms map in the same direction: a left arm is not mirrored, so the
        robot's hand goes where the operator's hand goes."""
        config = self.robot.config
        upper = unit(np.subtract(elbow, shoulder))
        forearm = unit(np.subtract(wrist, elbow))
        robot_elbow = self.shoulder+config['upper_arm_m']*upper
        return robot_elbow, robot_elbow+config['forearm_m']*forearm

    def palm_to_tool(self, palm_in_body, side):
        """Tool-site rotation that puts the LEAP palm in the operator's palm orientation."""
        palm = palm_frame_for_robot(palm_in_body, side)
        return palm@self.human_to_model.T@self.tool_to_palm.T

    def match_reference(self, anchor):
        """Robot joint configuration that best matches the operator's start posture."""
        home = self.robot.home.copy()
        if not self.match_start_pose:
            return home, {"matched": False, "reason": "match_start_pose disabled: reference = robot home"}
        config = self.robot.config
        elbow, wrist = self.posture_targets(anchor.shoulder_body_m, anchor.elbow_body_m, anchor.wrist_body_m)
        target = np.clip(wrist, config['workspace_lower_m'], config['workspace_upper_m'])
        orientation = self.palm_to_tool(anchor.palm_in_body, anchor.side)
        best = None
        seeds = [home]+[np.r_[home[:7]+offset, home[7:]] for offset in
                        ([0, 0, 0, 0, 1.2, 0, 0], [0, 0, 0, 0, -1.2, 0, 0], [0, -.4, 0, .6, 0, 0, 0], [0, .3, 0, -.3, 1.6, 0, 0])]
        for seed in seeds:
            seed = np.clip(seed, self.robot.lower, self.robot.upper)
            q, error, rotation_error = self.solve_arm(target, orientation, seed=seed, iterations=300, elbow_target=elbow,
                                                      min_iterations=300)
            elbow_error = float(np.linalg.norm(self.robot.forward(q).site('elbow').xpos-elbow))
            cost = error+ORIENTATION_WEIGHT*rotation_error+.3*elbow_error
            if best is None or cost < best[0]:
                best = (cost, q, error, rotation_error, elbow_error)
        _, q, error, rotation_error, elbow_error = best
        report = {"matched": True, "wrist_error_mm": error*1000, "palm_error_deg": float(np.degrees(rotation_error)),
                  "elbow_error_mm": elbow_error*1000, "workspace_limited": bool(np.linalg.norm(target-wrist) > 1e-6)}
        report["quality"] = "good" if error < .03 and rotation_error < np.radians(20) else "approximate"
        return q, report

    # ------------------------------------------------------------------ geometry
    def stabilized(self, points):
        """Shoulder/elbow/wrist rebuilt with the calibrated segment lengths.

        Single-view depth makes the measured segment lengths fluctuate from frame
        to frame; keeping the measured directions with constant lengths removes
        that jitter from the wrist position without delaying it."""
        points = np.asarray(points, float)
        if self.anchor is None:
            return points
        elbow = points[0]+self.anchor.upper_arm_model_m*unit(points[1]-points[0])
        wrist = elbow+self.anchor.forearm_model_m*unit(points[2]-points[1])
        return np.array([points[0], elbow, wrist])

    def desired_geometry(self, arm, hand):
        if self.anchor is None or self.human_open is None:
            raise ValueError("Open/neutral arm calibration is required")
        self.anchor.validate_measurement(arm)
        config = self.robot.config
        points = self.stabilized(arm['points_body_m'])
        displacement = points[2]-self.anchor_wrist
        requested = self.reference_tool+self.scale*displacement
        target = np.clip(requested, config['workspace_lower_m'], config['workspace_upper_m'])
        workspace_limited = bool(np.linalg.norm(target-requested) > 1e-9)
        if workspace_limited and self.policy == "reject":
            raise ValueError("Arm target is outside the configured robot workspace")
        palm_now = np.asarray(arm['basis_columns_in_model_camera']).T@np.asarray(hand['palm_basis_columns_in_model_camera'])
        side = self.anchor.side
        delta = palm_frame_for_robot(palm_now, side)@palm_frame_for_robot(self.anchor.palm_in_body, side).T
        orientation = delta@self.reference_rotation
        if self.match_start_pose:
            elbow, ghost_wrist = self.posture_targets(points[0], points[1], points[2])
        else:
            anchor_upper = unit(np.subtract(self.anchor.elbow_body_m, self.anchor.shoulder_body_m))
            anchor_forearm = unit(np.subtract(self.anchor.wrist_body_m, self.anchor.elbow_body_m))
            elbow = self.shoulder+config['upper_arm_m']*rotate_like(self.home_upper, anchor_upper, unit(points[1]-points[0]))
            ghost_wrist = elbow+config['forearm_m']*rotate_like(self.home_forearm, anchor_forearm, unit(points[2]-points[1]))
        return {"target": target, "requested": requested, "orientation": orientation, "elbow": elbow,
                "ghost_wrist": ghost_wrist, "workspace_limited": workspace_limited, "displacement": displacement}

    def solve_arm(self, target, orientation, seed=None, iterations=40, elbow_target=None, min_iterations=12):
        robot = self.robot
        q = self.seed.copy() if seed is None else np.asarray(seed, float).copy()
        lower, upper = robot.lower[:7], robot.upper[:7]
        jp, jr = np.zeros((3, robot.model.nv)), np.zeros((3, robot.model.nv))
        target, orientation = np.asarray(target, float), np.asarray(orientation, float)
        for iteration in range(iterations):
            data = robot.forward(q)
            site = data.site('tool')
            pos_error = target-site.xpos
            rot_error = rotation_vector(orientation@site.xmat.reshape(3, 3).T)
            converged = np.linalg.norm(pos_error) < .0007 and np.linalg.norm(rot_error) < .01
            if converged and (elbow_target is None or iteration >= min_iterations):
                break
            mujoco.mj_jacSite(robot.model, data, jp, jr, site.id)
            jac = np.vstack((jp[:, robot.dofs[:7]], ORIENTATION_WEIGHT*jr[:, robot.dofs[:7]]))
            error = np.r_[pos_error, ORIENTATION_WEIGHT*rot_error]
            # Adaptive damping: stronger far from the target and near singularities.
            damping = 2e-4+.02*min(1., float(np.linalg.norm(error)))**2
            step, free = limited_dls(jac, error, q[:7], lower, upper, damping, .2)
            if elbow_target is not None and free.sum() > 6:
                elbow = data.site('elbow')
                mujoco.mj_jacSite(robot.model, data, jp, jr, elbow.id)
                ej = jp[:, robot.dofs[:7]]*free
                jf = jac*free
                null = np.eye(7)-np.linalg.pinv(jf, rcond=.02)@jf
                swivel = null@ej.T@np.linalg.solve(ej@ej.T+.004*np.eye(3), np.asarray(elbow_target)-elbow.xpos)
                step = step+np.clip(.35*swivel, -.08, .08)
            q[:7] = np.clip(q[:7]+step, lower, upper)
        data = robot.forward(q)
        error = float(np.linalg.norm(data.site('tool').xpos-target))
        rotation_error = float(np.linalg.norm(rotation_vector(orientation@data.site('tool').xmat.reshape(3, 3).T)))
        return q, error, rotation_error

    # ------------------------------------------------------------------ hand
    def finger_joints(self, hand):
        """LEAP index/middle/ring joint targets from the operator's joint angles (motor order)."""
        q = np.zeros(16)
        open_joints = self.human_open['joints_deg']
        for finger in LONG_FINGERS:
            now, rest = hand['joints_deg'][finger], open_joints[finger]
            gain = self.gains[finger]
            flex = [gain[i]*np.radians(now[j]-rest[j]) for i, j in enumerate(("mcp", "pip", "dip"))]
            # Splay relative to the open pose (which removes the knuckle-fan bias of
            # the measurement). Positive LEAP abduction is toward the ring finger.
            # Splay is unreliable and collides on the LEAP once the finger bends, so fade it out.
            spread = -np.radians(now['splay']-rest['splay'])
            spread = float(np.clip(spread, -MAX_SPREAD_RAD, MAX_SPREAD_RAD))*max(0., 1-max(flex[0], 0)/1.1)
            q[list(FINGER_BLOCKS[finger])] = [spread, *flex]
        thumb, rest = hand['joints_deg']['thumb'], open_joints['thumb']
        q[14], q[15] = np.radians(thumb['mcp']-rest['mcp']), np.radians(thumb['ip']-rest['ip'])
        return np.clip(q, self.robot.lower[7:], self.robot.upper[7:])

    def thumb_target(self, hand):
        """Operator thumb tip relative to the thumb base, in LEAP palm coordinates.

        The direction from the thumb base to the tip is kept; its length is scaled
        from the operator's open-thumb length to the LEAP thumb's reach, so a
        curled thumb gives a proportionally shorter LEAP thumb.
        """
        mapping = self.human_to_model@np.diag([1., 1., -1. if self.anchor is not None and self.anchor.side == 'Left' else 1.])
        landmarks = np.asarray(hand['landmarks_palm_model_m'])
        return self.thumb_base+self.thumb_scale*(mapping@(landmarks[4]-landmarks[1]))

    def pinch(self, hand):
        """(finger, weight 0..1) for the fingertip the operator's thumb is closest to."""
        distances = {finger: hand['pinch_distance_palm_widths'][finger] for finger in LONG_FINGERS}
        finger = min(distances, key=distances.get)
        weight = float(np.clip((PINCH_START-distances[finger])/(PINCH_START-PINCH_FULL), 0, 1))
        return finger, weight

    def solve_hand(self, hand, seed=None, iterations=40):
        """Fingers from joint angles; thumb (and a pinching finger) by small IK. Returns q, fit dict."""
        robot = self.robot
        q = (self.seed if seed is None else np.asarray(seed, float)).copy()
        mapped = self.finger_joints(hand)
        q[7:19] = mapped[:12]
        thumb_target = self.thumb_target(hand)
        finger, weight = self.pinch(hand)
        prior_weights = np.ones(16)
        prior_weights[list(FINGER_BLOCKS[finger])] *= 1-.7*weight
        prior_weights[12:14], prior_weights[14:16] = .03, .08
        prior_values = mapped.copy()
        prior_values[12:14] = 0.
        lower, upper = robot.lower[7:], robot.upper[7:]
        jp, jr = np.zeros((3, robot.model.nv)), np.zeros((3, robot.model.nv))
        thumb_site, finger_site = robot.model.site('tip_thumb').id, robot.model.site('tip_'+finger).id
        for _ in range(iterations):
            data = robot.forward(q)
            root = data.body('leap_palm_lower')
            rotation = root.xmat.reshape(3, 3)
            thumb = rotation.T@(data.site_xpos[thumb_site]-root.xpos)
            tip = rotation.T@(data.site_xpos[finger_site]-root.xpos)
            mujoco.mj_jacSite(robot.model, data, jp, jr, thumb_site)
            thumb_jac = rotation.T@jp[:, robot.dofs[7:]]
            mujoco.mj_jacSite(robot.model, data, jp, jr, finger_site)
            finger_jac = rotation.T@jp[:, robot.dofs[7:]]
            residual = [THUMB_WEIGHT*(1-weight)*(thumb_target-thumb), THUMB_WEIGHT*weight*(tip-thumb),
                        prior_weights*(prior_values-q[7:])]
            jacobian = [THUMB_WEIGHT*(1-weight)*thumb_jac, THUMB_WEIGHT*weight*(thumb_jac-finger_jac), np.diag(prior_weights)]
            r, j = np.concatenate(residual), np.vstack(jacobian)
            step = np.linalg.solve(j.T@j+1e-3*np.eye(16), j.T@r)
            q[7:] = np.clip(q[7:]+np.clip(step, -.25, .25), lower, upper)
            if np.linalg.norm(step) < 1e-4:
                break
        data = robot.forward(q)
        root = data.body('leap_palm_lower')
        rotation = root.xmat.reshape(3, 3)
        thumb = rotation.T@(data.site('tip_thumb').xpos-root.xpos)
        tip = rotation.T@(data.site('tip_'+finger).xpos-root.xpos)
        goal = (1-weight)*thumb_target+weight*tip
        fit = {"thumb_fit_mm": float(np.linalg.norm(thumb-goal))*1000, "thumb_goal_palm": goal,
               "pinch_finger": finger if weight > 0 else None, "pinch_weight": weight,
               "pinch_gap_mm": float(np.linalg.norm(thumb-tip))*1000,
               "finger_tracking_error_rad": float(np.max(np.abs(q[7:19]-mapped[:12])))}
        return q, fit

    # ------------------------------------------------------------------ combined
    def joint_margins(self, q):
        """Distance (rad) of each arm joint from its nearest limit."""
        return np.minimum(q[:7]-self.robot.lower[:7], self.robot.upper[:7]-q[:7])

    def solve(self, arm, hand):
        geometry = self.desired_geometry(arm, hand)
        q, arm_error, rotation_error = self.solve_arm(geometry["target"], geometry["orientation"], elbow_target=geometry["elbow"])
        q, hand_fit = self.solve_hand(hand, q)
        if not np.isfinite(q).all():
            raise ValueError("Retargeting produced a non-finite solution")
        config = self.robot.config
        names = [name.replace('arm_', '').replace('_', ' ') for name in self.robot.names[:7]]
        margins = self.joint_margins(q)
        at_limit = [names[i] for i in range(7) if margins[i] <= 1e-4]
        near_limit = [f"{names[i]} ({np.degrees(margins[i]):.0f} deg left)" for i in range(7) if 1e-4 < margins[i] < NEAR_LIMIT_RAD]
        # Limits: the requested motion was constrained. Fit warnings: the robot
        # follows, but cannot match the human geometry exactly.
        reasons, warnings = [], []
        if geometry["workspace_limited"]:
            reasons.append("workspace boundary")
        if arm_error > config['arm_fit_tolerance_m']:
            reasons.append(f"reach ({arm_error*1000:.0f} mm short)")
        if at_limit:
            reasons.append("joint limit: "+", ".join(at_limit))
        if rotation_error > .35:
            warnings.append(f"wrist rotation off by {np.degrees(rotation_error):.0f} deg")
        if hand_fit["thumb_fit_mm"] > config['hand_fit_tolerance_m']*1000:
            warnings.append(f"thumb fit {hand_fit['thumb_fit_mm']:.0f} mm")
        guidance = ""
        if reasons or rotation_error > .35:
            # Tell the operator which way leads back toward the start pose.
            back = -np.asarray(geometry["displacement"])
            words = direction_words(back)
            guidance = ("Move your hand "+words+" toward the start pose") if words else ""
            if rotation_error > .35 or at_limit:
                guidance = (guidance+"; " if guidance else "")+"rotate your hand back toward the start orientation"
        metrics = {'arm_ik_error_mm': arm_error*1000, 'wrist_orientation_error_deg': float(np.degrees(rotation_error)),
                   'mean_fingertip_fit_error_mm': hand_fit["thumb_fit_mm"], 'thumb_fit_error_mm': hand_fit["thumb_fit_mm"],
                   'pinch': {'finger': hand_fit["pinch_finger"], 'weight': hand_fit["pinch_weight"],
                             'robot_gap_mm': hand_fit["pinch_gap_mm"]},
                   'translation_scale': self.scale, 'workspace_limited': geometry["workspace_limited"],
                   'limit_active': bool(reasons), 'limit_reasons': reasons, 'near_limits': near_limit,
                   'joint_margins_deg': dict(zip(names, np.degrees(margins).round(1).tolist())),
                   'fit_warnings': warnings, 'guidance': guidance, 'limit_policy': self.policy}
        if self.policy == "reject":
            if arm_error > config['arm_fit_tolerance_m'] or rotation_error > .35:
                raise ValueError(f"Arm IK cannot reach this pose (position error {arm_error*1000:.0f} mm)")
            if hand_fit["thumb_fit_mm"] > config['hand_fit_tolerance_m']*1000:
                raise ValueError(f"Thumb target cannot be matched (fit {hand_fit['thumb_fit_mm']:.0f} mm)")
        data = self.robot.forward(q)
        root = data.body('leap_palm_lower')
        rotation = root.xmat.reshape(3, 3)
        self.overlay = {"shoulder": self.shoulder.copy(), "elbow": geometry["elbow"], "ghost_wrist": geometry["ghost_wrist"],
                        "wrist": geometry["target"], "wrist_rotation": geometry["orientation"], "limit": bool(reasons),
                        "tips": [root.xpos+rotation@hand_fit["thumb_goal_palm"]]}
        self.seed = q.copy()
        return q, metrics

    def reference_overlay(self):
        """Ghost of the matched start pose, shown after the OPEN calibration."""
        if self.anchor is None:
            return None
        elbow, wrist = self.posture_targets(self.anchor.shoulder_body_m, self.anchor.elbow_body_m, self.anchor.wrist_body_m)
        return {"shoulder": self.shoulder.copy(), "elbow": elbow, "ghost_wrist": wrist, "wrist": self.reference_tool,
                "wrist_rotation": self.reference_rotation, "limit": False, "tips": []}


class SignalGenerator:
    def __init__(self, robot, max_age_ms=120):
        self.robot, self.max_age_ms = robot, max_age_ms
        self.last_q = self.clamp(robot.measured())
        self.last_time = None

    def clamp(self, q):
        return np.clip(np.asarray(q, float), self.robot.lower, self.robot.upper)

    def reconcile(self):
        # A measured joint can sit fractionally past its limit (contact, servo overshoot).
        self.last_q = self.clamp(self.robot.measured())
        self.last_time = None

    def generate(self, desired, source_ns, now_ns=None, speed_scale=1.):
        now_ns = time.perf_counter_ns() if now_ns is None else now_ns
        if not 0 <= (now_ns-source_ns)/1e6 <= self.max_age_ms:
            raise ValueError("Arm/hand source frame became stale before signal generation")
        desired = np.asarray(desired, float)
        if desired.shape != (23,) or not np.isfinite(desired).all():
            raise ValueError("Invalid coordinated arm/hand target")
        clamped = self.clamp(desired)
        if np.max(np.abs(clamped-desired)) > .05:
            raise ValueError("Joint target exceeds the robot model limits")
        dt = .02 if self.last_time is None else np.clip((now_ns-self.last_time)/1e9, 0, .05)
        velocity = speed_scale*np.r_[np.full(7, self.robot.config['max_velocity_rad_s']), np.full(16, self.robot.config['hand_max_velocity_rad_s'])]
        limited = self.clamp(self.last_q+np.clip(clamped-self.last_q, -velocity*dt, velocity*dt))
        self.last_q, self.last_time = limited.copy(), now_ns
        return {'schema': 'quadradex.control.v1', 'destination': 'simulation_only', 'hardware_enabled': False,
                'clock': 'host_perf_counter_ns', 'source_received_monotonic_ns': int(source_ns),
                'generated_monotonic_ns': int(now_ns), 'expires_monotonic_ns': int(source_ns+self.max_age_ms*1e6),
                'joint_names': self.robot.names, 'positions_rad': limited.tolist(),
                'arm_positions_rad': limited[:7].tolist(), 'leap_motor_ids': list(range(16)),
                'leap_sim_positions_rad': limited[7:].tolist(), 'leap_v1_sdk_positions_rad': leap_sim_to_sdk(limited[7:]).tolist(),
                'leap_model_revision': '150bc3d4b61fd6619193ba5a8ef209f3609ced89',
                'velocity_limited': bool(np.any(np.abs(limited-clamped) > 1e-7)), 'speed_scale': float(speed_scale)}
