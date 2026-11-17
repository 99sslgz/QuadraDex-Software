"""Latest-frame worker: body and hand results always share one source frame."""
from dataclasses import dataclass
import hashlib
from pathlib import Path
import sys
import threading
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/"cv_hand_tracking"))
from track_hand import ensure_model, select_hand
from hand_geometry import OneEuro, features
from teleoperation.arm_tracking import arm_features, pose_arrays, ArmContinuity

POSE_URL="https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
POSE_SHA="59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"


def ensure_models():
    hand=ensure_model(ROOT.parent/"cv_hand_tracking/models/hand_landmarker.task")
    pose=ROOT/"models/pose_landmarker_lite.task"
    if not pose.exists():
        print("Downloading body pose model...",flush=True)
        pose.parent.mkdir(parents=True,exist_ok=True)
        data=urllib.request.urlopen(POSE_URL,timeout=40).read()
        if hashlib.sha256(data).hexdigest()!=POSE_SHA:
            raise ValueError("Body model checksum mismatch")
        temporary=pose.with_suffix(".download")
        temporary.write_bytes(data)
        temporary.replace(pose)
    return pose,hand


def wrist_crop(pose_result, image, side):
    """Zoom the selected wrist for finger detail while keeping same-frame timing."""
    if not pose_result.pose_landmarks:
        return None
    points=pose_result.pose_landmarks[0]
    wrist,elbow=(16,14) if side=="right" else (15,13)
    if points[wrist].visibility is None or points[wrist].visibility<.5:
        return None
    height,width=image.shape[:2]
    w=np.array([points[wrist].x*width,points[wrist].y*height])
    e=np.array([points[elbow].x*width,points[elbow].y*height])
    # Include fingertips beyond the wrist, not just a wrist-centred square.
    center=w+.25*(w-e)
    span=np.clip(np.linalg.norm(w-e)*1.9,160,min(width,height)*.90)
    left,top=np.floor(center-span/2).astype(int)
    right,bottom=np.ceil(center+span/2).astype(int)
    left,top,right,bottom=max(0,left),max(0,top),min(width,right),min(height,bottom)
    return (left,top,right,bottom) if right-left>80 and bottom-top>80 else None


@dataclass
class Observation:
    frame: object
    completed_ns: int
    arm: dict|None
    hand: dict|None
    hand_image: np.ndarray|None
    pose_image: np.ndarray|None
    reason: str
    roi: tuple|None=None


class WholeArmVision:
    def __init__(self,camera,side="right",swap=False,min_visibility=.6):
        self.camera,self.side,self.swap,self.min_visibility=camera,side,swap,min_visibility
        self.latest=None
        self.error=None
        self.lock=threading.Lock()
        self.halt=threading.Event()
        pose,hand=ensure_models()
        self.paths=(pose,hand)
        self.thread=threading.Thread(target=self._work,daemon=True,name="synchronized-arm-hand")
        self.thread.start()

    def _work(self):
        filter_=OneEuro(3.,12.)
        arm_filter=OneEuro(3.,4.)
        continuity=ArmContinuity()
        physical_side=("left" if self.side=="right" else "right") if self.swap else self.side
        try:
            base=mp.tasks.BaseOptions
            vision=mp.tasks.vision
            pose_options=vision.PoseLandmarkerOptions(base_options=base(model_asset_path=str(self.paths[0])),running_mode=vision.RunningMode.VIDEO,num_poses=1,min_pose_detection_confidence=.5,min_pose_presence_confidence=.5,min_tracking_confidence=.5)
            hand_options=vision.HandLandmarkerOptions(base_options=base(model_asset_path=str(self.paths[1])),running_mode=vision.RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.5,min_tracking_confidence=.5)
            with vision.PoseLandmarker.create_from_options(pose_options) as pose, vision.HandLandmarker.create_from_options(hand_options) as hand:
                last_frame,last_ts=-1,-1
                while not self.halt.is_set():
                    frame=self.camera.get()
                    if frame is None or frame.index==last_frame:
                        self.halt.wait(.003)
                        continue
                    last_frame=frame.index
                    timestamp=max(last_ts+1,frame.received_ns//1_000_000)
                    last_ts=timestamp
                    rgb=cv2.cvtColor(frame.image,cv2.COLOR_BGR2RGB)
                    pose_result=pose.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),timestamp)
                    roi=wrist_crop(pose_result,frame.image,physical_side)
                    if roi:
                        x1,y1,x2,y2=roi
                        hand_rgb=np.ascontiguousarray(rgb[y1:y2,x1:x2])
                    else:
                        hand_rgb=rgb
                    hand_result=hand.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=hand_rgb),timestamp)
                    arm=measurement=hand_image=pose_image=None
                    reason=""
                    try:
                        pose_image,world,confidence=pose_arrays(pose_result)
                        arm=arm_features(pose_image,world,confidence,physical_side,self.min_visibility)
                        if arm_filter.time is not None and frame.received_ns/1e9-arm_filter.time>.25:
                            arm_filter=OneEuro(3.,4.)
                        world=arm_filter.update(world,frame.received_ns/1e9)
                        arm=arm_features(pose_image,world,confidence,physical_side,self.min_visibility)
                        # Expose the selected operator side, even for a mirrored camera.
                        arm["side"]=self.side.title()
                        continuity.check(arm,frame.received_ns/1e9)
                        selected,rejection=select_hand(hand_result,self.side.title(),self.swap,.6)
                        if selected is None or rejection:
                            raise ValueError(rejection or "Selected hand missing")
                        hand_image,hand_world,_,_=selected
                        if roi:
                            hand_image[:,0]=(hand_image[:,0]*(x2-x1)+x1)/rgb.shape[1]
                            hand_image[:,1]=(hand_image[:,1]*(y2-y1)+y1)/rgb.shape[0]
                            hand_image[:,2]*=(x2-x1)/rgb.shape[1]
                        wrist_id=16 if physical_side=="right" else 15
                        gap=np.linalg.norm((hand_image[0,:2]-pose_image[wrist_id,:2])*[rgb.shape[1],rgb.shape[0]])
                        if gap>max(70,np.linalg.norm((pose_image[wrist_id,:2]-pose_image[wrist_id-2,:2])*[rgb.shape[1],rgb.shape[0]])*.8):
                            raise ValueError("Detected hand does not match the selected arm wrist")
                        if filter_.time is not None and frame.received_ns/1e9-filter_.time>.25:
                            filter_=OneEuro(3.,12.)
                        measurement=features(filter_.update(hand_world,frame.received_ns/1e9))
                        reason="Synchronized shoulder / elbow / wrist / fingers"
                    except ValueError as error:
                        reason=str(error)
                        filter_=OneEuro(3.,12.)
                    observation=Observation(frame,time.perf_counter_ns(),arm,measurement,hand_image,pose_image,reason,roi)
                    with self.lock:
                        self.latest=observation
        except Exception as error:
            self.error=str(error)

    def get(self):
        with self.lock:
            return self.latest

    def close(self):
        self.halt.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("Inference worker did not stop; restart the program before changing cameras")
