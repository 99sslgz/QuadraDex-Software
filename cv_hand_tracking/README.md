# QuadraDex CV hand-tracking prototype

A runnable prototype of **S2 Hand acquisition**, with the input geometry needed for **S3 hand retargeting**. It uses a normal RGB webcam and MediaPipe's current Hand Landmarker Tasks API. It tracks continuous hand articulation rather than classifying a few gestures.

## Start on this Windows computer

From the repository root in PowerShell:

```powershell
.\cv_hand_tracking\run_tracker.bat
```

The launcher creates a local `.venv`, installs `requirements.txt`, and downloads the pinned Google model on first use. These generated dependencies are not stored in Git. Use 64-bit Python 3.11; this combination was tested with Python 3.11.5. No environment activation is required.

If you prefer running Python directly:

```powershell
cd cv_hand_tracking
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe track_hand.py
```

Startup opens a **camera picker** with the names of your connected cameras. Click a camera (or press 1-9), check its live preview, then click **Use camera** or press Enter. Only the selected camera is opened for preview. Press **B** if a camera needs the alternate Windows connection, or **R** to refresh the device list. The picker does not start tracking until you confirm your choice.

Keep the tracking window focused for keyboard controls. Hand selection defaults to **Auto**: show one hand; its identity is fixed for calibration and streaming. Before calibration, Auto can select whichever single hand you show. Press **H** or click **Hand** to cycle Auto / Right / Left. Press **M** if your camera driver mirrors the image and the detected left/right labels are reversed. Changing the hand or label convention clears calibration.

1. Place the camera on a fixed mount. Use even lighting and keep your entire hand, including the wrist, visible. Start around 40-70 cm from the lens, then adjust until the fingers occupy a useful part of the image without touching its edges.
2. Open and spread your hand naturally, then press **O** or click **1. Open**. The calibration card shows a one-second preparation countdown followed by a progress bar, percentage, and accepted sample count. Hold still until the card says the open pose is saved.
3. Close your fingers into a fist, bend your thumb as well, then press **C** or click **2. Closed**. Wait for the same countdown and progress display. The card shows which controlled fingers need more bend relative to the open pose. When complete, it reports that calibration was saved. An obstructed thumb may require a slight change in camera angle.
4. After calibration succeeds, press **Space** to start the geometric target stream. The display says `STREAMING`. Without `--udp`, targets are available in optional logs; the on-screen tracking works either way.
5. Move individual fingers, spread them, pinch, and rotate your palm. All five fingers are observed. The four LEAP target fingers are **thumb, index, middle, ring**, as specified in the concept design; the pinky is diagnostic only.
6. Press **S** to stop streaming, **Space** to start/resume, **R** to clear calibration in memory, or **Q/Esc** to quit. Press **V** or click **Camera** to return to the camera picker. Changing cameras stops the target stream and requires new calibration. Closing the window also exits. The software controls the data stream; these keys do not replace the robot's physical stop.

A brief tracking interruption **pauses calibration** and shows the reason while retaining progress. Missing frames never advance the bar. Unsteady finger estimates restart the stable sampling window with an explanation. Capture needs 1.25 seconds of accepted measurement time and at least eight samples after preparation; it times out visibly after 25 seconds. Only the four controlled fingers determine curl stability, so pinky movement does not cancel calibration. You can start O even before a hand is detected: the card waits and explains what is missing. C before O shows an explicit instruction to capture the open pose first.

The preview and target stream have separate freshness rules. A skeleton up to 250 ms old can remain visible in **amber** for diagnosis, with a `Preview only` or rejection message. Robot targets are still blocked as soon as their configured 100 ms age limit is exceeded. No hand positions are invented to fill missing frames. Wrong-hand selection, missing hands, cropped geometry and delayed inference are reported on screen instead of silently hiding every detection.

Calibration is saved as `calibration_right.json` or `calibration_left.json`, according to the hand actually selected. A new default launch asks for calibration again. To explicitly reuse and update a profile:

```powershell
.\run_tracker.bat --hand right --calibration calibration_right.json
```

Recalibrate when the operator changes or the profile no longer matches the hand. Loaded calibration never automatically starts streaming. If tracking fails during streaming, targets become unavailable and `TRACKING_LOST` latches. Restore valid tracking for at least 0.5 seconds and press Space to resume. This is local stream recovery; follower pose reconciliation belongs to S3/S4/S6.

## What is measured

| Output | Meaning |
| --- | --- |
| 21 image landmarks | Normalized image x/y and model-relative z; wrist and each finger joint/tip |
| 21 raw and filtered 3D landmarks | Model-estimated hand-relative coordinates in metres |
| Palm frame | Wrist origin; x from pinky MCP toward index MCP (radial); y toward middle MCP with the x component removed (distal); z = x cross y |
| Palm basis | Three basis columns expressed in the model's approximate camera axes; useful for relative palm orientation |
| Non-thumb joints | MCP out-of-palm bend, unsigned PIP/DIP bend, and signed in-palm splay, in degrees |
| Thumb joints | MCP/IP unsigned bends and a CMC bend proxy; full anatomical thumb opposition is not inferred from these scalars |
| Fingertips and distal directions | Position of each fingertip in the palm frame and direction of its last segment |
| Pinch distances | Thumb-to-other-tip distances divided by the current estimated palm width |
| Calibrated curl | 0 = the captured open pose; 1 = the captured closed pose, clamped to [0, 1] |
| Teleoperation geometry | Four fingertip positions, thumb-to-fingertip vectors and distal directions for geometric retargeting |

`joints_deg` contains human geometric estimates, **not LEAP joint angles**. For fingers, MCP bend is resolved from the palm plane separately from splay; PIP/DIP and thumb bends cannot distinguish flexion from hyperextension. Curl is a descriptive average of bends, normalized by the two calibration poses. It is not tendon displacement or force.

The palm frame is constructed separately for the selected hand and always forms a right-handed basis. Its z axis is defined by the cross product, not by a universal palmar-facing sign across both hands. Inference receives the camera image without additional mirroring. Only the preview is mirrored. If your camera driver mirrors its input and the selected physical hand is rejected as the opposite hand, use `--swap-handedness`. Verify the physical left/right selection before robot integration.

The One Euro filter attenuates small jitter and responds faster to larger movements. Validation happens on raw landmarks before filtering. Missing or implausible geometry resets the filter; a brief timing rejection does not constantly reset smoothing. Gaps longer than 250 ms reset it on the next accepted observation. Missing landmarks are never filled from the previous pose. Use `--no-smoothing` to compare raw behavior. Inspect raw and filtered data when assessing whether filtering hides small deliberate movements.

## Connect to the retargeting subsystem

In one PowerShell window, from this folder:

```powershell
.\.venv\Scripts\python.exe receive_targets.py
```

In another:

```powershell
.\run_tracker.bat --udp 127.0.0.1:5005 --log logs\trial_001.jsonl
```

Calibrate and press Space. The receiver prints calibrated curls for each new usable source frame. `receive_targets.py:consume_target()` is the documented integration point for your S3 retargeter. The receiver is a local diagnostic example; it does not connect to actuators.

Each UDP datagram is one UTF-8 JSON object using schema `quadradex.hand.v1`. The JSONL logger writes the same objects, one per line. A typical envelope has:

```text
session_id, sequence, source_frame_id, new_result, camera, calibration_progress
source_received_monotonic_ns, published_monotonic_ns
state, tracking_valid, calibrated, handedness, handedness_score
source_image_size_wh, timing, measurement, teleop_targets
```

`teleop_targets` is populated only when tracking is fresh, calibration is valid and the local stream has been started. It is `null` during calibration, when stopped, on invalid input, and during latched recovery. `measurement` and landmark arrays are also `null` whenever tracking is invalid. A final best-effort `STOPPED` packet inhibits targets on normal exit.

Within `teleop_targets`, the fingertip positions and thumb-to-tip vectors use **calibrated palm widths**, not robot metres. For example, `fingertips.index = [0.3, 1.6, -0.4]` means 0.3 reference palm widths radially, 1.6 distally and -0.4 along the cross-product normal, relative to the wrist. S3 must align and scale this geometry to the actual LEAP model, solve constrained inverse kinematics, and then pass the result to S4. The coordinate names and `finger_order` are explicit in each target.

The heartbeat defaults to 30 Hz; it may repeat a fresh observation. Repeated packets are **not additional measured hand states**. Use `source_frame_id` to reject duplicates and `sequence`/`session_id` to reject reordered packets and handle producer restarts. `new_result` indicates a newly processed inference completion, including completions that fail validation.

Both producer and example receiver must run on the **same computer**: freshness uses the high-resolution host performance counter (`clock = host_perf_counter_ns`). The existing timestamp field names retain `monotonic_ns` because this counter is monotonic. This avoids the approximately 16 ms `GetTickCount64` resolution observed with `time.monotonic_ns()` in this Windows Python 3.11 environment. The updated receiver also accepts legacy `host_monotonic_ns` packets using their original clock; use the updated receiver with the updated producer. The example binds to loopback and rejects future or old source times. Cross-computer operation requires a defined clock synchronization and transport scheme. The full landmark messages can span several network fragments; this transport is intended for local prototyping.

Before actuation, the downstream system must associate the original hand timestamp with the arm input, validate the complete arm/hand pair, reconcile the current follower pose on takeover, enforce robot joint/workspace/velocity limits, and inhibit execution on missing/newly stale inputs. A receiver watchdog is required even if the producer reports healthy tracking, since the producer can crash or UDP packets can disappear. The included receiver times out when no new valid source frame arrives; an actual controller needs the controlled stop and independent stop mechanisms in your concept design.

No LEAP motor order, joint signs, zero offsets or motion limits have been guessed. Those depend on the installed hand and its model. This prototype supplies the sensing interface; robot IK, robot commands, exoskeleton synchronization, and hardware safety remain separate integration work.

## Timing and settings

Default requests: **640 x 480 at 30 camera FPS**, 30 Hz observation/heartbeat publication, at least 60 Hz polling of completed results, 100 ms maximum target age, detection/presence/tracking thresholds 0.50, selected-handedness score at least 0.60. The detector thresholds now match the standard MediaPipe defaults; handedness confidence is still not a position-quality measure. Camera drivers can ignore requested settings; startup prints what the camera reports. The 100 ms freshness limit and 0.5 s recovery interval are prototype choices, not established project acceptance criteria.

The camera reader retains one latest frame. MediaPipe inference is asynchronous and can skip inputs while busy. Its callback retains one latest result; retained in-flight metadata is bounded. UDP sends are nonblocking, and file logging uses a bounded background queue. Packet counters expose locally dropped sends/log records; UDP delivery is not acknowledged. A stalled camera or inference result ages out, so a frozen process cannot create a valid stream through heartbeat alone.

Useful options:

```powershell
# Left hand, or an external camera
.\run_tracker.bat --hand left --camera 1

# List camera names and backend-specific indices without opening a camera
.\run_tracker.bat --list-cameras

# Return to the startup picker even when a camera index was supplied
.\run_tracker.bat --choose-camera --camera 1

# Alternative Windows camera backend
.\run_tracker.bat --backend dshow

# More image detail, if your machine can sustain the completed-result rate
.\run_tracker.bat --width 960 --height 720

# Compare unsmoothed tracking, or change the adaptive filter
.\run_tracker.bat --no-smoothing
.\run_tracker.bat --min-cutoff 3 --beta 15

# Diagnostic run without a preview; never automatically starts target streaming
.\run_tracker.bat --headless --duration 20 --log logs\diagnostic_001.jsonl

# Fetch/check model availability without opening the camera
.\run_tracker.bat --download-model
```

Existing log files are never overwritten. Choose a new filename per trial. Logs contain numerical hand data, not images/video. Disk-full or writer failures are reported and stop the producer.

`timing.inference_ms` measures host frame receipt to inference callback, including submission delay. `timing.result_age_ms` measures host frame receipt to packet construction; JSON encoding, delivery and downstream work happen afterwards. `camera_read_ms` is the blocking read duration. Ordinary OpenCV camera capture does not provide a reliable exposure timestamp: **host receipt is not sensor acquisition time**, and driver buffering may add hidden delay. The final p50/p95 summary covers accepted samples and is not a worst-case bound or a measurement of filter lag.

## Development verification

- Seventeen offline tests pass, including camera picker selection/preview handoff, named device discovery, auto hand selection, paused calibration progress, duplicate rejection, unstable-sample restart, timeout messaging, and the original checks: known bends, rigid-transform invariance, palm basis, individual finger motion/pinch, invalid geometry, jitter filtering, calibration validity and persistence, deliberate recovery, and stale receiver rejection.
- The actual downloaded model detects and passes geometry validation for both hands in Google's public sample photo.
- A 4.5-second integration run used repeated sample-photo frames plus a blank interval, real asynchronous inference and real loopback UDP. It exercised `READY -> STREAMING -> TRACKING_LOST -> STREAMING -> STOPPED`, checked that recovered tracking stayed inhibited until explicit restart, and verified logs and received packets. Synthetic calibration and simulated key presses were test fixtures, not a measured operator calibration.
- A second real-model integration run switched cameras during streaming and verified an immediate inhibit packet, a new source session, cleared calibration, blocked targets on the new camera and closure of both camera readers. Camera input and key presses were simulated for this repeatable check.
- Historical first-version webcam checks found repeated age rejections; the original 960 x 720 run had 28 stale-output records in 106 records. The revised code uses the high-resolution counter, polls results more often, and keeps diagnostic preview independent from target permission. Both connected devices were enumerated correctly as **Integrated Webcam** and **EMEET SmartCam C60E 4K**, and both opened at 640 x 480 / 30 camera FPS. The second-camera diagnostic saw no hand in view; no tracking-rate claim is made for that run. These short checks do not establish reliable tracking for your poses or verify PR-01/PR-02.
- The camera picker and calibration progress/paused displays were visually inspected using the public sample image. A human open/closed calibration session and physical LEAP tracking have not been verified here.

Run the offline checks:

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```

## What to evaluate for the capstone prototype

Use fixed camera placement and log separate trials for an open palm, slow single-finger bends, spread, thumb opposition, pinch, a fist, wrist rotation, fast motion, hand exit/re-entry and exoskeleton/object occlusion. Measure distinct valid source-frame rate, invalid-frame fraction, gaps between usable observations, jitter when stationary, and lag under motion. Keep lighting/distance/pose conditions with each trial. Do not count repeated heartbeat packets as fresh updates.

Your baseline requires sensor-to-command latency at most 150 ms (PR-01), average command updates at least 20 Hz (PR-02), and mean desired-versus-achieved follower fingertip error at most 25 mm (PR-04). The logger helps characterize the sensing contribution. Verify PR-01 at real sensor acquisition and follower command issuance, PR-02 at the follower output, and PR-04 with the actual robot and an independent fingertip measurement procedure. Webcam landmark confidence does not establish any of those outcomes.

Single-view inferred depth is ambiguous. Crossed fingers, a closed fist, fingers behind an object, motion blur, and views from the edge of the hand can produce plausible but wrong estimates. The gross geometry checks cannot identify every such error. Thumb opposition and small depth changes need particular attention in your P1/P3 studies. Use measured accuracy and occlusion results to decide whether a second calibrated view or depth sensor is needed. Absolute wrist translation in the robot workspace still comes from your S1 exoskeleton, not these hand-centred 3D landmarks. No contact or force sensing is provided.

## Sources and implementation files

- Project baseline: `../docs/QuadraDex_spec_v1.1.pdf`, especially C-01/C-03, FR-03/04/22/23/28 and PR-01/02/04.
- Concept: `../docs/Concept_Design_Quadradex (1).pdf`, Sections 6.2, 6.3 and 6.8 (S2, S3 and I2); Section 11 (P1/P3/P4).
- [Google Hand Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python): API, model and landmark coordinates.
- [Google Python example and sample image](https://colab.research.google.com/github/google-ai-edge/mediapipe-samples/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb).
- `track_hand.py`: webcam, asynchronous inference, preview, calibration controls, UDP and logging.
- `hand_geometry.py`: frame geometry, adaptive filtering, features, calibration and stream state.
- `receive_targets.py`: local consumer example with freshness checks and a timeout.
- `camera_picker.py`: named device discovery, asynchronous preview opening, and device selection.
- `calibration_ui.py`: guided capture progress, preparation, pauses and timeout messages.
- `test_hand_tracking.py`: camera-independent checks.

Model: Google `hand_landmarker/float16/1/hand_landmarker.task`, downloaded over HTTPS and checksum-checked on download. SHA-256: `fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1`.

Camera discovery uses [cv2-enumerate-cameras](https://github.com/lukehugh/cv2_enumerate_cameras), with the same explicit Windows backend for device enumeration and capture so displayed indices agree with the opened device.
