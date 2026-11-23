"""Kinematic stand-in for a human right hand, producing MediaPipe-style landmarks.

Used by the tests and the --demo mode to drive the real hand-retargeting code
with known human finger angles. Dimensions are typical adult values; this is a
test fixture, not a measurement of any operator.

Palm coordinates: x radial (toward the thumb), y distal (toward the fingers),
z palmar. For a right hand this frame is right-handed, matching
hand_geometry.palm_frame (radial x distal = palmar).
"""
import numpy as np

MCP = {"index": (.032, .088), "middle": (.010, .092), "ring": (-.011, .088), "pinky": (-.030, .078)}
SEGMENTS = {"index": (.042, .025, .021), "middle": (.047, .029, .022), "ring": (.044, .027, .021),
            "pinky": (.034, .020, .019)}
BASE = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
THUMB_CMC = np.array([.020, .025, .008])
THUMB_SEGMENTS = (.046, .034, .028)
OPEN_THUMB = {"abduction": 15., "spread": 42., "mcp": 8., "ip": 6.}
FIST_THUMB = {"abduction": 45., "spread": 12., "mcp": 55., "ip": 45.}


def _unit(v):
    return v/np.linalg.norm(v)


def _thumb(params):
    ab, sp = np.radians(params["abduction"]), np.radians(params["spread"])
    metacarpal = _unit(np.array([np.cos(ab)*np.sin(sp), np.cos(ab)*np.cos(sp), np.sin(ab)]))
    # The thumb flexes toward the palm centre: ulnar and palmar.
    bend = np.array([-1., -.2, .8])
    bend = _unit(bend-np.dot(bend, metacarpal)*metacarpal)
    points = [THUMB_CMC]
    angle = 0.
    for length, flex in zip(THUMB_SEGMENTS, (0., params["mcp"], params["ip"])):
        angle += np.radians(flex)
        points.append(points[-1]+length*(np.cos(angle)*metacarpal+np.sin(angle)*bend))
    return points


def hand_landmarks(fingers=None, thumb=None):
    """21 x 3 landmarks in palm coordinates.

    fingers: {name: (splay_deg, mcp_deg, pip_deg, dip_deg)}; missing fingers are straight.
    Positive splay is toward the thumb. thumb: dict like OPEN_THUMB.
    """
    fingers = fingers or {}
    points = np.zeros((21, 3))
    thumb_points = _thumb({**OPEN_THUMB, **(thumb or {})})
    points[1:5] = thumb_points
    z = np.array([0., 0., 1.])
    for name, (x, y) in MCP.items():
        splay, mcp, pip, dip = fingers.get(name, (0., 0., 0., 0.))
        longitudinal = np.array([0., 1., 0.]) if name != "pinky" else _unit(np.array([-.12, 1., 0.]))
        lateral = np.cross(longitudinal, z)  # toward the thumb
        # Straight fingers are parallel; splay rotates within the palm plane.
        direction = np.cos(np.radians(splay))*longitudinal+np.sin(np.radians(splay))*lateral
        base = BASE[name]
        points[base] = [x, y, 0.]
        angle = 0.
        for k, (length, flex) in enumerate(zip(SEGMENTS[name], (mcp, pip, dip))):
            angle += np.radians(flex)
            points[base+k+1] = points[base+k]+length*(np.cos(angle)*direction+np.sin(angle)*z)
    return points


def to_camera(local, palm_in_camera, origin=(0., 0., .5)):
    """Place palm-coordinate landmarks in MediaPipe camera axes (x right, y down, z away)."""
    return np.asarray(local)@np.asarray(palm_in_camera).T+np.asarray(origin)


# Right hand raised, palm toward the camera, fingers up: radial -> camera +x
# (the operator's left), distal -> up (camera -y), palmar -> toward the camera (-z).
PALM_TOWARD_CAMERA = np.column_stack(([1., 0., 0.], [0., -1., 0.], [0., 0., -1.]))


def pinch_landmarks(finger="index", finger_flex=(0., 60., 65., 35.)):
    """Thumb tip placed on the given fingertip (a pinch), found by a small numerical fit."""
    fingers = {finger: finger_flex}
    target = hand_landmarks(fingers)[BASE[finger]+3]
    params = dict(FIST_THUMB)
    keys = ("abduction", "spread", "mcp", "ip")
    x = np.array([params[k] for k in keys], float)
    for _ in range(60):
        tip = _thumb(dict(zip(keys, x)))[-1]
        error = target-tip
        if np.linalg.norm(error) < 1e-4:
            break
        jac = np.zeros((3, 4))
        for i in range(4):
            dx = x.copy()
            dx[i] += .5
            jac[:, i] = (_thumb(dict(zip(keys, dx)))[-1]-tip)/.5
        x += np.clip(np.linalg.lstsq(jac, error, rcond=None)[0], -8, 8)
        x = np.clip(x, [-10, -40, 0, 0], [80, 70, 90, 90])
    thumb = dict(zip(keys, x))
    points = hand_landmarks(fingers, thumb)
    return points, thumb
