"""Shoulder/elbow/wrist estimates and a torso-relative operator calibration."""
from dataclasses import dataclass, asdict
import numpy as np


def unit(vector):
    length = np.linalg.norm(vector)
    if length < 1e-7:
        raise ValueError("Degenerate arm or torso geometry")
    return vector / length


def pose_arrays(result):
    if len(result.pose_landmarks) != 1 or len(result.pose_world_landmarks) != 1:
        raise ValueError("Show one person, with shoulders, hips, elbow and wrist visible")
    image = np.array([[p.x,p.y,p.z] for p in result.pose_landmarks[0]])
    world = np.array([[p.x,p.y,p.z] for p in result.pose_world_landmarks[0]])
    confidence = np.array([min(p.visibility if p.visibility is not None else 0., p.presence if p.presence is not None else 0.) for p in result.pose_landmarks[0]])
    return image,world,confidence


def arm_features(image, world, confidence, side="right", min_visibility=.6):
    if image.shape!=(33,3) or world.shape!=(33,3) or confidence.shape!=(33,):
        raise ValueError("Incomplete body landmarks")
    if not np.isfinite(image).all() or not np.isfinite(world).all() or not np.isfinite(confidence).all():
        raise ValueError("Non-finite body landmarks")
    shoulder, elbow, wrist = (12,14,16) if side=="right" else (11,13,15)
    required = [11,12,23,24,shoulder,elbow,wrist]
    if np.min(confidence[required]) < min_visibility:
        raise ValueError("Arm/torso obscured: keep both shoulders, hips, elbow and wrist visible")
    if np.any(image[required,:2]<0) or np.any(image[required,:2]>1):
        raise ValueError("Move back: part of the arm or torso is outside the image")
    upper = float(np.linalg.norm(world[elbow]-world[shoulder]))
    forearm = float(np.linalg.norm(world[wrist]-world[elbow]))
    width = float(np.linalg.norm(world[11]-world[12]))
    if not .12<upper<.65 or not .10<forearm<.60 or not .15<width<.70:
        raise ValueError("Implausible estimated arm lengths; improve the camera view")
    left = unit(world[11]-world[12])
    up = (world[11]+world[12]-world[23]-world[24])/2
    up = unit(up-np.dot(up,left)*left)
    forward = unit(np.cross(left,up))
    basis = np.column_stack((forward,left,up))
    local = (world[[shoulder,elbow,wrist]]-world[shoulder])@basis
    flexion = np.degrees(np.arccos(np.clip(np.dot(unit(world[elbow]-world[shoulder]),unit(world[wrist]-world[elbow])),-1,1)))
    return {"side":side.title(),"frame":"shoulder_origin_forward_left_up",
            "shoulder_model_m":world[shoulder].tolist(),"elbow_model_m":world[elbow].tolist(),"wrist_model_m":world[wrist].tolist(),
            "points_body_m":local.tolist(),"basis_columns_in_model_camera":basis.tolist(),
            "upper_arm_model_m":upper,"forearm_model_m":forearm,"shoulder_width_model_m":width,
            "elbow_flexion_deg":float(flexion),"minimum_visibility":float(np.min(confidence[required])),
            "accuracy_status":"not_independently_validated"}


@dataclass
class ArmAnchor:
    side: str
    wrist_body_m: list
    elbow_body_m: list
    palm_in_body: list
    upper_arm_model_m: float
    forearm_model_m: float

    @classmethod
    def capture(cls, arms, hands):
        if len(arms)<8 or len(arms)!=len(hands):
            raise ValueError("Need at least eight synchronized arm/hand calibration samples")
        wrists=np.array([a["points_body_m"][2] for a in arms])
        if np.max(np.linalg.norm(wrists-np.median(wrists,axis=0),axis=1))>.035:
            raise ValueError("Arm moved during the OPEN calibration; hold it still and retry O")
        palm = np.asarray(arms[-1]["basis_columns_in_model_camera"]).T @ np.asarray(hands[-1]["palm_basis_columns_in_model_camera"])
        return cls(arms[-1]["side"],np.median(wrists,axis=0).tolist(),np.median([a['points_body_m'][1] for a in arms],axis=0).tolist(),palm.tolist(),
                   float(np.median([a["upper_arm_model_m"] for a in arms])),float(np.median([a["forearm_model_m"] for a in arms])))

    def validate_measurement(self, arm):
        if arm["side"]!=self.side:
            raise ValueError("The calibrated arm side changed")
        for name in ("upper_arm_model_m","forearm_model_m"):
            if abs(arm[name]/getattr(self,name)-1)>.25:
                raise ValueError("Arm length estimate changed; restore visibility or recalibrate")

    def to_dict(self):
        return asdict(self)


class ArmContinuity:
    def __init__(self):
        self.time=None
        self.wrist=None

    def check(self, arm, source_s):
        wrist=np.asarray(arm["points_body_m"][2])
        if self.time is not None and 0<source_s-self.time<.25:
            if np.linalg.norm(wrist-self.wrist)/(source_s-self.time)>3.5:
                self.time,self.wrist=None,None
                raise ValueError("Sudden arm estimate jump; restore a clear view")
        self.time,self.wrist=source_s,wrist
