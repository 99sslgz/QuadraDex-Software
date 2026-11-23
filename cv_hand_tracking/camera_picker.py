"""Named camera discovery and a responsive, single-camera preview picker."""
from dataclasses import dataclass
import queue
import sys
import threading
import time

import cv2
import numpy as np


PICKER_WINDOW = "QuadraDex | Choose a camera"
BACKENDS = {"auto": cv2.CAP_ANY, "msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}


@dataclass(frozen=True)
class CameraChoice:
    index: int
    backend: str
    name: str


def discover_cameras(backend="auto"):
    from cv2_enumerate_cameras import enumerate_cameras
    # Explicit backend makes the displayed index agree with VideoCapture.
    selected = "msmf" if backend == "auto" and sys.platform == "win32" else backend
    devices = enumerate_cameras(BACKENDS[selected])
    return [CameraChoice(device.index, selected, device.name) for device in devices]


def fit_image(image, width, height):
    result = np.full((height, width, 3), (29, 32, 38), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    y, x = (height - resized.shape[0]) // 2, (width - resized.shape[1]) // 2
    result[y:y+resized.shape[0], x:x+resized.shape[1]] = resized
    return result


def picker_view(devices, selected, frame=None, status="Opening preview...", backend="msmf", title="Choose your tracking camera"):
    view = np.full((700, 1160, 3), (24, 27, 32), dtype=np.uint8)

    def text(line, xy, scale=.55, color=(230, 233, 238)):
        cv2.putText(view, line, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    text(title, (28, 43), .9)
    text("Select a device on the left, check its live preview, then choose Use camera.", (28, 76))
    text(f"Camera connection: {backend.upper()}   |   B: try other connection", (28, 111), .48)
    for row, device in enumerate(devices[:9]):
        top = 142 + row * 49
        cv2.rectangle(view, (22, top), (419, top+43), (64, 91, 75) if row == selected else (40, 44, 51), -1)
        name = f"{row+1}. {device.name}"
        text(name[:41], (34, top+27), .52)
    if not devices:
        text("No cameras found. Press B or R.", (30, 178), .55, (110, 190, 255))
    if frame is not None:
        view[142:142+430, 440:440+692] = fit_image(cv2.flip(frame.image, 1), 692, 430)
        text("LIVE PREVIEW", (455, 168), .5, (105, 230, 175))
    else:
        cv2.rectangle(view, (440, 142), (1132, 572), (34, 38, 44), -1)
        text("Waiting for camera preview", (565, 344), .65)
    text(status[:83], (441, 601), .5, (110, 195, 255))
    cv2.rectangle(view, (870, 628), (1130, 677), (64, 117, 86) if frame is not None else (60, 62, 66), -1)
    text("Use camera  [Enter]", (886, 660), .63)
    text("Click / 1-9: select    Up/Down: browse    R: refresh    Esc: cancel", (28, 665), .48)
    return view


def choose_camera(camera_factory, args, title="Choose your tracking camera", exclude=()):
    """Return (choice, open_reader), or None. Never selects without user input.

    exclude: camera indices already in use (for example the first of two cameras)."""
    backend = "msmf" if args.backend == "auto" and sys.platform == "win32" else args.backend

    def discover(backend):
        return [device for device in discover_cameras(backend) if device.index not in exclude]

    devices = discover(backend)
    selected = 0
    current = None
    pending = queue.Queue()
    cancel = threading.Event()
    generation = 0
    status = "Choose a camera to see its preview"
    clicks = []
    opening_since = time.perf_counter()

    def mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicks.append((x, y))

    def start_preview():
        nonlocal generation, current, status, opening_since
        generation += 1
        token = generation
        if current is not None:
            current.close()
            current = None
        if not devices:
            return
        choice = devices[selected]
        opening_since = time.perf_counter()
        status = f"Opening {choice.name}..."

        def open_device():
            reader = None
            try:
                reader = camera_factory(choice.index, args.width, args.height, args.camera_fps, choice.backend)
                # Ownership transfer is protected against picker cancellation.
                with transfer_lock:
                    if cancel.is_set():
                        reader.close()
                    else:
                        pending.put((token, reader, None))
            except Exception as error:
                with transfer_lock:
                    if not cancel.is_set():
                        pending.put((token, None, str(error)))

        threading.Thread(target=open_device, daemon=True, name="camera-preview-open").start()

    transfer_lock = threading.Lock()
    cv2.namedWindow(PICKER_WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(PICKER_WINDOW, mouse)
    start_preview()
    try:
        while True:
            while True:
                try:
                    token, reader, error = pending.get_nowait()
                except queue.Empty:
                    break
                if token != generation:
                    if reader is not None:
                        reader.close()
                else:
                    current = reader
                    status = error or "Waiting for the first frame..."
            frame = current.get() if current is not None else None
            if frame is not None and (time.perf_counter_ns() - frame.received_ns) / 1e9 > 1:
                frame = None
                status = "Camera preview stopped. Try R or another device."
            if current is not None and current.error:
                frame, status = None, current.error
            elif frame is not None:
                status = f"{devices[selected].name} | {frame.image.shape[1]} x {frame.image.shape[0]} | preview ready"
            elif time.perf_counter() - opening_since > 6:
                status = "No preview yet. Close other camera apps, choose another camera, or press B."
            cv2.imshow(PICKER_WINDOW, picker_view(devices, selected, frame, status, backend, title))
            key = cv2.waitKeyEx(15)
            if cv2.getWindowProperty(PICKER_WINDOW, cv2.WND_PROP_VISIBLE) < 1 or key in (27, ord("q")):
                return None
            changed = False
            for x, y in clicks:
                row = (y - 142) // 49
                if 22 <= x <= 419 and y >= 142 and 0 <= row < min(9, len(devices)):
                    changed, selected = row != selected, row
                if 870 <= x <= 1130 and 628 <= y <= 677:
                    key = 13
            clicks.clear()
            if ord("1") <= key <= ord("9") and key - ord("1") < len(devices):
                changed, selected = selected != key - ord("1"), key - ord("1")
            elif key in (2490368, 65362, ord("k")) and devices:
                changed, selected = True, (selected - 1) % len(devices)
            elif key in (2621440, 65364, ord("j")) and devices:
                changed, selected = True, (selected + 1) % len(devices)
            elif key in (ord("r"), ord("b")):
                if key == ord("b") and sys.platform == "win32":
                    backend = "dshow" if backend != "dshow" else "msmf"
                devices, selected, changed = discover(backend), 0, True
            if changed:
                start_preview()
            elif key in (10, 13) and frame is not None:
                chosen = current
                current = None  # Transfer the already-open reader to the tracker.
                return devices[selected], chosen
    finally:
        with transfer_lock:
            cancel.set()
            while not pending.empty():
                _, reader, _ = pending.get_nowait()
                if reader is not None:
                    reader.close()
        if current is not None:
            current.close()
        cv2.destroyWindow(PICKER_WINDOW)
