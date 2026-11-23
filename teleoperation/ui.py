"""Camera / calibration window for the full-arm teleoperation app."""
import textwrap

import cv2
import numpy as np

from hand_geometry import CONNECTIONS, FINGERS

CAM_W, CAM_H = 960, 540
STRIP_H = 300
PANEL_W = 460
WIDTH, HEIGHT = CAM_W+PANEL_W, CAM_H+STRIP_H
FONT = cv2.FONT_HERSHEY_SIMPLEX

BG = (30, 27, 25)
CARD = (46, 42, 39)
INK = (232, 232, 232)
MUTED = (160, 160, 160)
GOOD = (110, 220, 150)
BAD = (90, 110, 245)
WARN = (60, 190, 250)
ACCENT = (235, 170, 60)

STATE_COLOURS = {"STREAMING": GOOD, "READY": ACCENT, "CALIBRATING": WARN, "TRACKING_HOLD": WARN,
                 "LIMIT_ACTIVE": BAD, "TRACKING_LOST": BAD, "CALIBRATION_REQUIRED": WARN, "STOPPED": MUTED,
                 "ALIGNING": ACCENT, "FAULT": BAD, "CAMERA_ERROR": BAD, "STEREO_SETUP": ACCENT}
STATE_TEXT = {"STREAMING": "TELEOPERATING (simulation)", "READY": "READY - press Space to start",
              "CALIBRATING": "CALIBRATING", "TRACKING_HOLD": "BRIEF TRACKING GAP - holding pose",
              "LIMIT_ACTIVE": "LIMIT ACTIVE - motion constrained", "TRACKING_LOST": "TRACKING LOST - press Space to resume",
              "CALIBRATION_REQUIRED": "CALIBRATION REQUIRED", "STOPPED": "STOPPED",
              "ALIGNING": "ALIGNING - robot moving to your start pose", "FAULT": "ERROR RECOVERED - robot holding",
              "CAMERA_ERROR": "CAMERA PROBLEM - robot holding, retrying",
              "STEREO_SETUP": "TWO-CAMERA SETUP - move both arms slowly"}

# (key, label, x1, y1, x2, y2) in window pixels, inside the right panel.
BUTTONS = [("o", "1.Open O", 0, 0, 0, 0), ("c", "2.Fist C", 0, 0, 0, 0), (" ", "Start/Stop", 0, 0, 0, 0),
           ("v", "Camera V", 0, 0, 0, 0), ("h", "Arm L/R H", 0, 0, 0, 0), ("m", "Mirror M", 0, 0, 0, 0),
           ("k", "Stereo K", 0, 0, 0, 0), ("r", "Reset R", 0, 0, 0, 0), ("b", "Objects B", 0, 0, 0, 0),
           ("q", "Quit Q", 0, 0, 0, 0)]


def _layout_buttons():
    out = []
    width, gap, x0 = 82, 6, CAM_W+14
    for i, (key, label, *_rest) in enumerate(BUTTONS):
        row, col = divmod(i, 5)
        x1 = x0+col*(width+gap)
        y1 = HEIGHT-98+row*46
        out.append((key, label, x1, y1, x1+width, y1+38))
    return out


BUTTONS = _layout_buttons()
UPPER_BODY = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24), (0, 11), (0, 12)]


def button_at(x, y):
    for key, _label, x1, y1, x2, y2 in BUTTONS:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return ord(key)
    return None


def _text(img, text, xy, scale=.5, colour=INK, thick=1):
    cv2.putText(img, text, (int(xy[0]), int(xy[1])), FONT, scale, colour, thick, cv2.LINE_AA)


def _to_display(points, mirror=True):
    """Normalized work-image coordinates -> camera-area pixels (selfie view)."""
    pts = np.asarray(points, float)[..., :2].copy()
    if mirror:
        pts[..., 0] = 1-pts[..., 0]
    return np.rint(pts*[CAM_W, CAM_H]).astype(int)


def _arrow_to_start(view, observation, side, start_wrist, active):
    """Ring at the wrist position captured in the start pose; arrow back to it when needed."""
    if start_wrist is None:
        return
    target = tuple(_to_display(start_wrist))
    colour = BAD if active else (150, 150, 150)
    cv2.circle(view, target, 22, colour, 2, cv2.LINE_AA)
    _text(view, "START", (target[0]-24, target[1]-28), .45, colour, 1)
    if active and observation.pose_image is not None:
        wrist = tuple(_to_display(observation.pose_image[16 if side == "right" else 15]))
        if np.hypot(wrist[0]-target[0], wrist[1]-target[1]) > 30:
            cv2.arrowedLine(view, wrist, target, BAD, 4, cv2.LINE_AA, tipLength=.12)


def _camera_area(observation, side, valid, age_ms):
    frame = observation.frame.image
    view = cv2.resize(cv2.flip(frame, 1), (CAM_W, CAM_H), interpolation=cv2.INTER_AREA)
    if age_ms > 300:
        cv2.putText(view, "Waiting for fresh tracking...", (30, 60), FONT, .8, WARN, 2, cv2.LINE_AA)
        return view
    chain = (12, 14, 16) if side == "right" else (11, 13, 15)
    if observation.pose_image is not None:
        p = _to_display(observation.pose_image)
        for a, b in UPPER_BODY:
            cv2.line(view, tuple(p[a]), tuple(p[b]), (150, 150, 150), 2, cv2.LINE_AA)
        arm_colour = GOOD if observation.arm is not None else WARN
        for a, b in zip(chain, chain[1:]):
            cv2.line(view, tuple(p[a]), tuple(p[b]), arm_colour, 5, cv2.LINE_AA)
        for index, name in zip(chain, ("S", "E", "W")):
            cv2.circle(view, tuple(p[index]), 9, arm_colour, -1, cv2.LINE_AA)
            _text(view, name, p[index]+[-5, 5], .42, (20, 20, 20), 1)
        other = 11 if side == "right" else 12
        cv2.circle(view, tuple(p[other]), 7, (170, 170, 170), -1, cv2.LINE_AA)
    else:
        # Framing guide when no person is found.
        cv2.rectangle(view, (CAM_W//2-230, 40), (CAM_W//2+230, CAM_H-30), WARN, 2)
        _text(view, "Fit your head, both shoulders and the selected arm in view", (CAM_W//2-300, 30), .6, WARN, 2)
    if observation.roi:
        x1, y1, x2, y2 = observation.roi
        h, w = frame.shape[:2]
        a = _to_display([[x2/w, y1/h], [x1/w, y2/h]])
        cv2.rectangle(view, tuple(a[0]), tuple(a[1]), (200, 160, 110), 1, cv2.LINE_AA)
    if observation.hand_image is not None:
        hp = _to_display(observation.hand_image)
        colour = (255, 220, 120) if observation.hand is not None else (120, 140, 255)
        for a, b in CONNECTIONS:
            cv2.line(view, tuple(hp[a]), tuple(hp[b]), colour, 2, cv2.LINE_AA)
        for point in hp:
            cv2.circle(view, tuple(point), 3, (60, 200, 255), -1, cv2.LINE_AA)
    return view


def _stereo_area(observation, ui):
    """Both camera views side by side, and a fused 3D front/top view of the tracked arm and hand."""
    area = np.full((CAM_H, CAM_W, 3), (24, 22, 21), np.uint8)
    names = ui.get("camera_names") or ["Camera 1", "Camera 2"]
    for i, view in enumerate(observation.views):
        x0 = i*(CAM_W//2)
        if view is None:
            _text(area, f"{names[i]}: waiting for a matching frame", (x0+20, 140), .5, WARN)
            continue
        tile = cv2.resize(_camera_area(view, ui["side"], ui["valid"], ui["age_ms"]), (CAM_W//2, CAM_H//2), interpolation=cv2.INTER_AREA)
        area[:CAM_H//2, x0:x0+CAM_W//2] = tile
        _text(area, f"{i+1}: {names[i][:30]}", (x0+8, 20), .45, INK, 1)
    cv2.line(area, (CAM_W//2, 0), (CAM_W//2, CAM_H//2), (60, 60, 60), 2)
    _fused_view(area, observation, ui)
    return area


def _fused_view(area, observation, ui):
    """Front (x-y) and top (x-z) projections of the fused 3D arm + hand, camera-1 axes."""
    top = CAM_H//2+8
    for k, title in enumerate(("FUSED 3D - front view", "FUSED 3D - top view (camera 1 at bottom)")):
        x0 = 12+k*(CAM_W//2)
        cv2.rectangle(area, (x0, top), (x0+CAM_W//2-24, CAM_H-8), (34, 31, 29), -1)
        _text(area, title, (x0+8, top+18), .42, ACCENT)
    arm, hand = observation.arm, observation.hand
    if arm is None:
        _text(area, (observation.arm_reason or "arm not fused")[:60], (24, top+60), .45, WARN)
        return
    chain = np.array([arm["shoulder_model_m"], arm["elbow_model_m"], arm["wrist_model_m"]])
    points = [chain]
    if hand is not None:
        basis = np.asarray(hand["palm_basis_columns_in_model_camera"])
        points.append(np.asarray(chain[2])+np.asarray(hand["landmarks_palm_model_m"])@basis.T)
    centre = chain[0]
    for k, axes in enumerate(((0, 1), (0, 2))):
        x0 = 12+k*(CAM_W//2)
        origin = np.array([x0+(CAM_W//2-24)/2, (top+CAM_H-8)/2+20])
        scale = 260.

        def to_px(xyz):
            xy = (np.asarray(xyz)[..., list(axes)]-centre[list(axes)])
            if k == 1:
                xy = xy*[1, -1]   # top view: depth (z) upward = away from camera 1
            xy = xy*[-1, 1]       # selfie orientation like the camera tiles
            return np.rint(origin+xy*scale).astype(int)
        arm_px = to_px(points[0])
        for a, b in zip(arm_px, arm_px[1:]):
            cv2.line(area, tuple(a), tuple(b), GOOD, 4, cv2.LINE_AA)
        for p in arm_px:
            cv2.circle(area, tuple(p), 6, GOOD, -1, cv2.LINE_AA)
        if len(points) > 1:
            hand_px = to_px(points[1])
            for a, b in CONNECTIONS:
                cv2.line(area, tuple(hand_px[a]), tuple(hand_px[b]), (255, 220, 120), 1, cv2.LINE_AA)
    sources = (observation.fusion or {}).get("pose_sources") or {}
    text = "  ".join(f"{name}: {src.replace(' cameras', '').replace('camera ', 'cam')}" for name, src in sources.items())
    _text(area, text[:110], (20, CAM_H-16), .4, MUTED)


def choose_mode(window="QuadraDex | camera mode"):
    """Ask for one or two cameras. Returns 1, 2, or None (cancelled)."""
    clicks = []

    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, mouse)
    boxes = {1: (40, 120, 440, 380), 2: (480, 120, 880, 380)}
    try:
        while True:
            view = np.full((440, 920, 3), BG, np.uint8)
            _text(view, "How many cameras are you using?", (40, 60), .9, INK, 2)
            _text(view, "Click a box or press 1 / 2.   Esc: cancel", (40, 92), .5, MUTED)
            texts = {1: ["ONE CAMERA (1)", "Camera in front of you.", "Simple setup. Overlapping arm/hand", "poses can be ambiguous."],
                     2: ["TWO CAMERAS (2)", "Front camera + side camera", "(45-90 deg apart). Triangulated 3D;", "handles overlap and folding."]}
            for mode, (x1, y1, x2, y2) in boxes.items():
                cv2.rectangle(view, (x1, y1), (x2, y2), CARD, -1)
                cv2.rectangle(view, (x1, y1), (x2, y2), ACCENT, 2)
                for i, line in enumerate(texts[mode]):
                    _text(view, line, (x1+24, y1+50+i*44), .75 if i == 0 else .55, ACCENT if i == 0 else INK, 2 if i == 0 else 1)
            cv2.imshow(window, view)
            key = cv2.waitKey(20) & 255
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1 or key in (27, ord('q')):
                return None
            if key in (ord('1'), ord('2')):
                return key-ord('0')
            for x, y in clicks:
                for mode, (x1, y1, x2, y2) in boxes.items():
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        return mode
            clicks.clear()
    finally:
        cv2.destroyWindow(window)


def _bar(img, x, y, width, fraction, colour, height=8):
    cv2.rectangle(img, (x, y), (x+width, y+height), (70, 66, 62), -1)
    cv2.rectangle(img, (x, y), (x+int(width*float(np.clip(fraction, 0, 1))), y+height), colour, -1)


def render(observation, ui):
    """ui: dict with state, reason, note, side, mirrored, camera_name, progress, calibrated, opened,
    metrics (dict), fit (dict|None), travel (dict), age_ms, valid."""
    canvas = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
    side = ui["side"]
    if observation is not None and getattr(observation, "views", None):
        canvas[:CAM_H, :CAM_W] = _stereo_area(observation, ui)
    elif observation is not None:
        view = _camera_area(observation, side, ui["valid"], ui["age_ms"])
        _arrow_to_start(view, observation, side, ui.get("start_wrist"), ui["state"] in ("LIMIT_ACTIVE",) or bool(ui.get("guidance")))
        canvas[:CAM_H, :CAM_W] = view
    else:
        _text(canvas, "Starting arm + finger tracking...", (40, CAM_H//2), .8)
    # Bottom strip: state, reason, instruction, calibration progress.
    strip_y = CAM_H
    state = ui["state"]
    colour = STATE_COLOURS.get(state, INK)
    cv2.rectangle(canvas, (0, strip_y), (CAM_W, HEIGHT), (36, 33, 31), -1)
    cv2.rectangle(canvas, (0, strip_y), (8, HEIGHT), colour, -1)
    _text(canvas, STATE_TEXT.get(state, state), (24, strip_y+38), .85, colour, 2)
    for i, line in enumerate(textwrap.wrap(ui["reason"] or "", 95)[:2]):
        _text(canvas, line, (24, strip_y+68+i*22), .5, MUTED)
    progress = ui["progress"]
    instruction = progress["message"] if progress.get("stage") else ui["note"]
    if state == "STEREO_SETUP":
        instruction = ("Stand where BOTH cameras see your head, shoulders and arms. Move both arms slowly in all directions, "
                       "including toward and away from the cameras, until the bar is full.")
    for i, line in enumerate(textwrap.wrap(instruction or "", 80)[:3]):
        _text(canvas, line, (24, strip_y+122+i*26), .62, ACCENT if progress.get("stage") else INK, 1)
    stereo = ui.get("stereo") or {}
    if state == "STEREO_SETUP":
        _bar(canvas, 24, strip_y+204, CAM_W-60, stereo.get("progress", 0), GOOD, 14)
    if progress.get("stage"):
        _bar(canvas, 24, strip_y+204, CAM_W-60, progress.get("progress", 0), GOOD, 14)
        _text(canvas, f"{round(100*progress.get('progress', 0))}%  {progress.get('accepted_samples', 0)} samples  {progress.get('phase', '').upper()}",
              (CAM_W-330, strip_y+196), .45, MUTED)
    # Limit / alignment guidance: what to do, in large type.
    guidance = ui.get("guidance") or ""
    near = ui.get("near_limits") or []
    if guidance:
        cv2.rectangle(canvas, (16, strip_y+214), (CAM_W-16, HEIGHT-8), (40, 40, 95), -1)
        for i, line in enumerate(textwrap.wrap("DO THIS: "+guidance, 72)[:2]):
            _text(canvas, line, (28, strip_y+242+i*30), .62, INK, 2)
    elif near and state in ("STREAMING", "ALIGNING"):
        _text(canvas, ("Approaching joint limit: "+", ".join(near))[:110], (24, strip_y+246), .55, WARN, 1)
    elif stereo.get("warning"):
        _text(canvas, stereo["warning"][:110], (24, strip_y+246), .55, WARN, 1)
    elif ui.get("partial"):
        _text(canvas, ui["partial"][:110], (24, strip_y+246), .55, WARN, 1)
    elif ui.get("warning"):
        _text(canvas, ui["warning"][:110], (24, strip_y+246), .5, WARN, 1)
    # Right panel.
    x = CAM_W+16
    _text(canvas, "QUADRADEX  ARM + HAND", (x, 32), .7, INK, 2)
    _text(canvas, (ui["camera_name"] or "Camera")[:44], (x, 56), .42, MUTED)
    if stereo.get("mode") == 2:
        quality = stereo.get("quality")
        detail = (f"stereo {quality} ({stereo.get('rms_px', 0):.1f} px, {stereo.get('angle_deg', 0):.0f} deg)" if quality else "stereo setup needed")
        live = stereo.get("live_px")
        if live is not None:
            detail += f" live {live:.1f}px"
        _text(canvas, f"Arm: {side.title()}   2 cameras: {detail}"[:58], (x, 80), .44, WARN if quality == "poor" else INK)
    else:
        _text(canvas, f"Arm: {side.title()}   1 camera   Mirror fix: {'on' if ui['mirrored'] else 'off'}", (x, 80), .48)
    y = 100
    cv2.rectangle(canvas, (CAM_W+8, y), (WIDTH-8, y+196), CARD, -1)
    _text(canvas, "WHAT THE CAMERA SEES", (x, y+24), .5, ACCENT)
    rows = list(observation.parts) if observation is not None and observation.parts else []
    hand_ok = observation is not None and observation.hand is not None
    hand_seen = observation is not None and observation.hand_image is not None
    rows += [("Hand found", hand_seen, None, "" if hand_seen else "open your hand toward the camera"),
             ("Finger tracking", hand_ok, None, "" if hand_ok else (observation.hand_reason if observation is not None else ""))]
    for i, (label, ok, score, hint) in enumerate(rows):
        yy = y+50+i*24
        cv2.circle(canvas, (x+8, yy-5), 7, GOOD if ok else BAD, -1, cv2.LINE_AA)
        _text(canvas, label, (x+24, yy), .48)
        detail = f"{score:.2f}" if score is not None else ""
        if hint and (not ok or "camera" in hint):   # two cameras: show where each joint came from
            detail = (detail+"  " if detail else "")+hint
        _text(canvas, detail[:38], (x+170, yy), .42, MUTED if ok else WARN)
    y += 206
    cv2.rectangle(canvas, (CAM_W+8, y), (WIDTH-8, y+92), CARD, -1)
    _text(canvas, "CALIBRATION", (x, y+24), .5, ACCENT)
    _text(canvas, "1. Start pose + open hand: "+("SAVED" if ui["opened"] else "press O"), (x, y+50), .48,
          GOOD if ui["opened"] else INK)
    _text(canvas, "2. Fist: "+("SAVED" if ui["calibrated"] else "press C"), (x, y+76), .48, GOOD if ui["calibrated"] else INK)
    travel = ui.get("travel") or {}
    if travel and not ui["calibrated"]:
        _text(canvas, " ".join(f"{k}:{'ok' if v else 'bend'}" for k, v in travel.items()), (x+200, y+76), .4, MUTED)
    y += 102
    checks = ui.get("pose_checks") or []
    reference = ui.get("reference") or {}
    cv2.rectangle(canvas, (CAM_W+8, y), (WIDTH-8, y+128), CARD, -1)
    _text(canvas, "START POSE", (x, y+24), .5, ACCENT)
    if reference.get("matched"):
        _text(canvas, f"robot match: {reference.get('quality')} ({reference.get('wrist_error_mm', 0):.0f} mm, "
                      f"{reference.get('palm_error_deg', 0):.0f} deg)", (x+120, y+24), .4, GOOD if reference.get("quality") == "good" else WARN)
    else:
        _text(canvas, "upper arm down, forearm up, palm to camera", (x+120, y+24), .4, MUTED)
    for i, (label, ok, detail) in enumerate(checks[:4]):
        yy = y+48+i*21
        cv2.circle(canvas, (x+8, yy-5), 6, GOOD if ok else WARN, -1, cv2.LINE_AA)
        _text(canvas, label, (x+24, yy), .44)
        _text(canvas, detail[:34], (x+190, yy), .4, MUTED if ok else WARN)
    y += 138
    cv2.rectangle(canvas, (CAM_W+8, y), (WIDTH-8, y+122), CARD, -1)
    _text(canvas, "FINGERS (curl)", (x, y+22), .5, ACCENT)
    hand = observation.hand if observation is not None else None
    for i, name in enumerate(FINGERS):
        yy = y+42+i*17
        _text(canvas, name, (x, yy), .4)
        if hand:
            _bar(canvas, x+70, yy-8, 330, hand["curl_raw"][name], GOOD, 7)
    y += 132
    m = ui["metrics"]
    fit = ui.get("fit")
    lines = [f"Paired results {m.get('valid_hz', 0):.0f}/s   age {ui['age_ms']:.0f} ms",
             f"Inference: pose {m.get('pose_ms', 0):.0f} ms, hand {m.get('hand_ms', 0):.0f} ms"]
    if fit:
        lines.append(f"Wrist IK {fit['arm_ik_error_mm']:.0f} mm  rot {fit['wrist_orientation_error_deg']:.0f} deg  thumb {fit['mean_fingertip_fit_error_mm']:.0f} mm")
        if fit.get("limit_reasons"):
            lines.append("Limit: "+"; ".join(fit["limit_reasons"]))
    for i, line in enumerate(lines[:3]):
        _text(canvas, line[:60], (x, y+i*21), .43, WARN if line.startswith("Limit") else MUTED)
    for key, label, x1, y1, x2, y2 in BUTTONS:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (72, 80, 70), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (95, 105, 92), 1)
        _text(canvas, label, (x1+5, y1+24), .38)
    return canvas
