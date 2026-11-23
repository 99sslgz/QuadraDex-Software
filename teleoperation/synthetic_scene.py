"""Synthetic operator seen by two virtual cameras, for testing the two-camera mode.

Produces per-view Observation objects with the same raw fields the real camera
workers provide (normalized 2D landmarks, confidences, per-view 3D estimates),
from a known 3D body + hand and known camera poses. Test fixture only.
"""
import time

import numpy as np

from teleoperation import synthetic_hand as synth
from teleoperation.arm_tracking import part_status
from teleoperation.stereo import project, rodrigues
from teleoperation.vision import Observation
from track_hand import Frame

SIZE = (1280, 720)


def camera(hfov_deg):
    focal = SIZE[0]/(2*np.tan(np.radians(hfov_deg)/2))
    return np.array([[focal, 0, SIZE[0]/2], [0, focal, SIZE[1]/2], [0, 0, 1.]])


class TwoCameraScene:
    """Camera 1 in front of the operator; camera 2 to the operator's right-front, turned toward them."""

    def __init__(self, angle_deg=60., hfov=(66., 72.), distance=1.7, noise_px=1.2, seed=0):
        self.K = [camera(hfov[0]), camera(hfov[1])]
        self.distance = distance
        centre = np.array([0., 0., distance])
        a = np.radians(angle_deg)
        # Camera 2 sits on a circle around the operator, turned about the vertical (camera y) axis.
        self.R2 = rodrigues([0., -a, 0.])
        position = centre+np.array([-np.sin(a), 0., -np.cos(a)])*distance
        self.t2 = -self.R2@position
        self.noise = noise_px
        self.rng = np.random.default_rng(seed)
        self.index = 0
        self.occluded = {1: set(), 2: set()}   # pose ids, or 'hand', hidden in a view

    def body(self, t, side="right", curl=None):
        """33 pose landmarks + 21 hand landmarks (camera-1 axes, metres) at time t."""
        z = self.distance
        X = np.zeros((33, 3))
        X[0] = [0, -.27, z-.08]
        for i in range(1, 11):
            X[i] = [(-1)**i*.03*((i+1)//2), -.30+.01*(i % 3), z-.06]
        X[11], X[12] = [.19, 0, z], [-.19, 0, z]
        X[23], X[24] = [.15, .52, z], [-.15, .52, z]
        for s, sign in ((12, -1), (11, 1)):
            elbow = X[s]+[sign*.03+.04*np.sin(t*.9), .27+.03*np.cos(t*.7), -.08-.05*np.sin(t*.5)]
            forearm = np.array([.06*np.sin(t*.8), -.17+.08*np.sin(t*.6), -.17-.07*np.cos(t*.4)])
            e, w = (14, 16) if s == 12 else (13, 15)
            X[e], X[w] = elbow, elbow+.25*forearm/np.linalg.norm(forearm)
        wrist = X[16] if side == "right" else X[15]
        curl = .5*(1-np.cos(t*.8)) if curl is None else curl
        thumb = {k: (1-curl)*synth.OPEN_THUMB[k]+curl*synth.FIST_THUMB[k] for k in synth.OPEN_THUMB}
        local = synth.hand_landmarks({f: (0, 80*curl, 95*curl, 55*curl) for f in synth.MCP}, thumb)
        if side == "left":
            local = local*[-1, 1, 1]
        tilt = rodrigues([.3*np.sin(t*.5), .4*np.sin(t*.3), 0.])
        hand = (local-local[0])@(tilt@synth.PALM_TOWARD_CAMERA).T+wrist
        return X, hand

    def observe(self, t, side="right", received_ns=None, curl=None):
        X, hand = self.body(t, side, curl)
        received_ns = time.perf_counter_ns() if received_ns is None else received_ns
        self.index += 1
        out = []
        for view, (K, R, tt) in enumerate(((self.K[0], np.eye(3), np.zeros(3)), (self.K[1], self.R2, self.t2)), start=1):
            uv, _ = project(K, R, tt, X)
            uvh, _ = project(K, R, tt, hand)
            uv = uv+self.rng.normal(0, self.noise, uv.shape)
            uvh = uvh+self.rng.normal(0, self.noise, uvh.shape)
            image = np.column_stack((uv/SIZE, np.zeros(33)))
            confidence = np.full(33, .95)
            for i in self.occluded[view]:
                if isinstance(i, int):
                    confidence[i] = .05
            centre = (X[23]+X[24])/2
            world = (X-centre)@R.T          # per-view model estimate: metric, hip-centred, this camera's axes
            seen = 'hand' not in self.occluded[view]
            hand_image = np.column_stack((uvh/SIZE, np.zeros(21))) if seen else None
            hand_world = (hand-hand.mean(axis=0))@R.T if seen else None
            frame = Frame(self.index, np.zeros((72, 128, 3), np.uint8), received_ns, 0.)
            out.append(Observation(frame, received_ns, None, None, hand_image, image, "synthetic", None,
                                   part_status(image, confidence, side), "", "" if seen else "hand hidden", None,
                                   {"pose": 1., "hand": 1.}, confidence, world, hand_world, SIZE))
        return out
