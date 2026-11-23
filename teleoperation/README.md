# Full-arm tracking, LEAP signals and simulation

This experimental branch adds synchronized camera tracking of a selected shoulder, elbow, wrist and all finger landmarks. A configurable seven-joint demo robot arm carries the official **right LEAP Hand V1** model on a bench with the project's evaluation objects. The simulation follows generated position-control signals.

The physical arm has not been selected and the physical LEAP revision is unconfirmed. The demo arm is an adjustable model, not a representation of purchased hardware. The V1 model follows the project's original LEAP baseline. This application generates V1-compatible signal values and drives the simulator; it does not connect to a motor bus or move physical actuators.

## Run

From the repository root, with 64-bit Python 3.11 installed:

```powershell
.\run_teleop.bat
```

The launcher installs dependencies into `cv_hand_tracking/.venv`. The body and hand inference models download once and stay outside Git (the default body model is MediaPipe Pose Landmarker **Full**; `--pose-model lite|heavy` selects another). The LEAP meshes, URDF, MIT license and pinned source manifest are included in the repository.

At start-up choose **one camera** or **two cameras** (or pass `--cameras 1|2`). Two windows open: a camera/calibration window and an interactive MuJoCo robot simulation. Select the camera(s) in the named camera picker; with two cameras you pick the front camera first, then the side camera. The simulation initially holds its home pose.

**Two cameras (recommended when the arm and hand overlap or fold).** Put camera 1 in front of you, roughly upright, and camera 2 to one side, **45-90 degrees** around you, both 1.2-2 m away and both seeing your head, shoulders and the whole selected arm. Do not move either camera during a session. The first time (and whenever the cameras change) a **two-camera setup** runs automatically (`STEREO_SETUP`): stand where both cameras see you and move both arms slowly in all directions, including toward and away from the cameras, until the bar is full (about 15 s); the geometry is then computed in about a second. No printed target is needed: your own body landmarks, seen by both cameras, calibrate the pair. The result is saved to `teleoperation/stereo_calibration.json` (not in Git) and reused automatically while the same two cameras are used at the same resolution. The header shows its quality ("good"/"usable"/"poor", reprojection error, angle between the cameras). Press **K** to redo it (for example after a camera was bumped; a warning appears when the two views stop agreeing). The camera area shows both views and a fused 3D front/top view; "What the camera sees" shows, for each joint, whether it came from both cameras or only one.

1. **Framing.** Sit or stand about 1.2-2 m from the camera so that your **head, both shoulders and the whole selected arm (shoulder to fingertips)** are visible. Hips are *not* required, so a desk in front of you is fine. Keep the camera roughly upright and keep one person in view. The default request is 1280 x 720 at 30 camera FPS.
2. **Check the panel "What the camera sees".** Every row should be green: shoulder, other shoulder, elbow, wrist, hand found, finger tracking. A red row names what is missing (for example "elbow outside the image"). The selected arm is drawn with S-E-W markers on the camera view. Select Right or Left with **H** / **Arm L/R**. If the highlighted arm is the wrong one, your camera driver delivers a mirrored image: press **M** / **Mirror**.
3. **Step 1 - start pose + open hand (O).** The recommended start pose is: **upper arm relaxed down at your side, elbow bent so the forearm points up and forward, open hand with the palm facing the camera and the fingers pointing up** (like raising your hand). It gives the camera a clear view of every finger. The **START POSE** card checks it live (upper arm down, elbow bent, palm toward camera, fingers up) and says what to adjust; the checks are guidance, not a gate. Press **O**. After a short countdown, hold still until the progress bar finishes. If the arm or fingers move, the stable window restarts automatically; a missing measurement pauses it and the panel says why. The robot then computes a **reference pose that matches your measured start posture** (upper-arm direction, elbow bend and palm orientation) and shows it as a ghost in the 3D window; the card reports how closely the robot can match it.
4. **Step 2 - fist (C).** Make a fist with the thumb bent and press **C**. Only the hand needs to be tracked for this step. Each finger joint's open-to-fist range is used to scale that joint, so your fist produces a full LEAP fist. Insufficient finger travel gives an explicit retry instruction.
5. **Teleoperate (Space).** Return to the start pose; with 0.5 s of continuous arm + finger tracking press **Space**. The robot first moves to the matched reference pose at reduced speed (`ALIGNING`, `align_speed_scale`), then follows you: the wrist follows your wrist's displacement from the start pose (scaled by robot/human arm length), the palm follows your palm rotation, the elbow follows your upper-arm direction, the index/middle/ring fingers follow your own finger joint angles and the thumb follows your thumb tip, snapping onto a LEAP fingertip when you pinch. Pinky landmarks remain diagnostic.
6. **Limits.** When you move beyond what the robot can do, the state turns `LIMIT_ACTIVE`, the robot keeps executing the closest feasible command, and both windows say why (workspace boundary, reach, which joint is at its limit) and **what to do** ("Move your hand up and forward toward the start pose; rotate your hand back..."). A ring on the camera view marks where your wrist was in the start pose, with an arrow back to it. Joints within about 7 degrees of a limit are listed as "approaching joint limit" before the limit is reached. Returning toward the start pose clears the state automatically.
7. **Stopping and recovery.** **S** or **Space** stops the paired arm/hand stream. If only the hand or only the arm drops out for up to 0.8 s (`--part-hold-ms`), the missing part holds its last measurement and the other keeps following (the strip says "fingers holding their last pose" or "arm holding"); control does not freeze. If both are missing, a gap shorter than 0.5 s (`--loss-grace-ms`) shows `TRACKING_HOLD`: no command is sent, the simulated robot holds its pose and streaming continues when fresh tracking returns. A longer gap latches `TRACKING_LOST`; press **Space** to resume after recovery. **R** clears calibration, **V** / **Camera** reopens camera selection, **M** / **N** fix a mirrored camera 1 / camera 2, **K** redoes the two-camera setup, **B** puts the objects back. Camera/side changes inhibit targets and require new calibration.
8. **The program only exits when you ask.** **Q/Esc** or closing a window ends the session; nothing else does. A camera that stops delivering frames gives `CAMERA_ERROR`: the robot holds and the camera is reopened automatically every 3 s (or press **V** to pick another). An unexpected internal error gives `FAULT`: the robot holds, the error is shown and printed to the console, and **Space** resumes once tracking is valid (**R** resets). A log-file write problem is shown as a warning and logging stops; the session continues. Only start-up problems (invalid configuration, missing models, the simulator window failing to open) can end the program before the session begins.

In the 3D window: drag to orbit, scroll to zoom, **[** / **]** cycle the fixed cameras (operator view, hand close-up, side, top), **F8** follows the hand and **F9** resets the objects. The overlay shows your arm posture at robot scale (orange), the commanded wrist pose (green sphere with axes, orange-red at a limit) and the thumb target (green dot), plus state, fit numbers, pinch status, limit reasons and the "Do this" guidance.

Arm/hand calibration is session-local. It is deliberately not loaded from the original hand-only calibration file: an arm anchor and a camera/side association must be established together.

## Try the simulator without a camera

```powershell
.\run_teleop.bat --demo
```

This produces clearly labeled **synthetic demo** signals: a generated wrist path through the same arm IK, finger motion, signal generator and simulator. It demonstrates simulator motion and signal conventions, not camera accuracy. A finite run with numerical logs and a robot-only image:

```powershell
.\run_teleop.bat --demo --headless --duration 8 --log teleoperation\logs\demo_001.jsonl --snapshot teleoperation\reports\demo.png
```

Use a new log filename per run. Camera images/video are not saved by the logger; `--snapshot` renders the simulated robot only.

Other options:

```powershell
.\run_teleop.bat --list-cameras
.\run_teleop.bat --cameras 1 --camera 0
.\run_teleop.bat --cameras 2 --camera 0 --camera2 1
.\run_teleop.bat --cameras 2 --camera 0 --camera2 1 --mirrored2
.\run_teleop.bat --camera 1 --side left
.\run_teleop.bat --camera 1 --backend dshow
.\run_teleop.bat --pose-model heavy
.\run_teleop.bat --mirrored
.\run_teleop.bat --download-models
.\run_teleop.bat --arm-config teleoperation\demo_arm.json
.\run_teleop.bat --udp 127.0.0.1:5006 --log teleoperation\logs\trial_001.jsonl
```

The original `cv_hand_tracking/run_tracker.bat` remains a separate hand-only tool.

## What the system measures

MediaPipe Pose Landmarker (Full by default, VIDEO mode) provides the shoulder, elbow and wrist estimates. A square window around the selected hand is cropped from the same frame, resized to 288 x 288 and passed to Hand Landmarker (VIDEO mode), so the fingers get enough pixels. The window **stays still while the hand moves inside it** and only re-centres when the hand nears its edge; when the hand is lost, the search window grows. (A window that moved every frame kept shifting the hand inside the model's input, which broke MediaPipe's own frame-to-frame hand tracking and caused frequent re-detection, jumps and dropouts.) **Both models process the same captured frame and preserve the same host-receipt timestamp.** Crop coordinates are mapped back into the full camera image.

The hand is associated with the arm by **distance between the hand's wrist landmark and the pose wrist**, never by MediaPipe's Left/Right hand label. On an unmirrored camera, the hand label is the opposite of the pose model's anatomical side (MediaPipe's hand classifier assumes a mirrored selfie image), which made the previous label check reject the selected hand. If the driver mirrors the image, **M** un-mirrors it before inference so that both models see a normal image.

Only both shoulders and the selected elbow and wrist are required. Hips are not used. The body frame has its origin at the shoulder midpoint, x forward (toward the camera), y toward the operator's left and z up. Heading comes from the horizontal shoulder line and "up" from the camera's vertical, so torso yaw does not move the robot. The inference worker reads only the latest frame, and the UI reads only the latest completed pair. Visibility/presence, image bounds, plausible proportions, temporal wrist continuity (a single implausible jump is dropped, a confirmed relocation is accepted) and hand geometry are checked before paired control becomes available.

**Stability measures.** Once the arm is tracked, it stays tracked down to 60 % of the visibility threshold (hysteresis), so a landmark hovering around the threshold no longer flickers in and out. MediaPipe's tracking thresholds are lowered slightly so it keeps tracking through brief uncertainty instead of re-detecting. A hand is accepted if it is near the pose wrist *or* near where the accepted hand was in the previous frame (the pose wrist is often a few centimetres off when the hand is in front of the body), and fingertips slightly past the image edge are tolerated. The arm segments are rebuilt with the calibrated upper-arm/forearm lengths (measured directions kept), which removes single-view depth noise in segment length from the wrist position. A briefly missing hand or arm is bridged by its last measurement (above). One-Euro filters reduce the remaining jitter; they do not fill in an occluded arm.

**Two cameras.** Each camera runs the same per-view pipeline. Observations from the two cameras are paired by capture time (closest pair within 50 ms; otherwise only the newer view is used). Each needed landmark is triangulated when both views see it confidently and the views agree (reprojection error below max(12 px, 3x the setup's error)). When one view loses a landmark - typically when the arm or fingers overlap or fold in that view - it is rebuilt from the other camera's viewing ray, placed with the bone direction from the already-fused parent joint (shoulder -> elbow -> wrist) as seen by that camera, or the previous fused position. The hand uses triangulated landmarks when they form a plausible hand; otherwise the finger shape comes from the camera that sees the palm most squarely, rotated into the common frame. All fused points are in camera-1 axes, so everything downstream (body frame, calibration, retargeting, limits) is the same as with one camera. The two-camera setup estimates the relative camera pose from the essential matrix of the body/hand correspondences (RANSAC) and refines rotation, translation direction and both focal lengths by robust Levenberg-Marquardt (weak prior: 70 degree horizontal field of view); lens distortion is neglected and metric scale comes from your shoulder width as estimated by MediaPipe. A scale error of a few percent changes only the overall size of motion, which is normalised by your measured arm length anyway.

Visibility is an estimate of observability, **not proof of metric accuracy**. With one camera, inferred depth, self-occlusion, sleeve/object occlusion and wrist axial rotation remain uncertain, and depth toward the camera is the least reliable axis. Two cameras address depth and single-view occlusion but rely on the setup quality, both cameras staying put, and on MediaPipe finding the same anatomical points in both views. The palm provides the wrist orientation estimate. This CV extension does not replace the specification's encoder-equipped leader exoskeleton for formal verification without an agreed requirement change and validation evidence.

## Mapping and control signals

`robot_model.py` builds a MuJoCo scene from the configurable demo arm and the pinned LEAP URDF. Published LEAP joint transforms, limits, mesh placements and inertias are used directly; only colours/materials are assigned for display. Fingertip sites are reproducible distal-surface estimates computed from the published STL meshes; they are not measured contact pads. The hand is mounted palm-down with the fingers along the forearm.

**Start-pose matching.** At the OPEN calibration the robot solves for the configuration whose upper arm, forearm and LEAP palm point the same way as yours in your body frame (forward / left / up = the robot base frame), using the same IK with a 300-iteration converged elbow term. While streaming, the wrist target is that reference wrist plus your scaled wrist displacement since calibration, and the palm target is your palm rotation since calibration applied to the reference palm; the elbow target follows your current upper-arm direction. Following relative to the *achieved* reference keeps motion consistent even when your start posture can only be approximated. `"match_start_pose": false` restores the previous behaviour (reference = robot home).

**Left arm.** The LEAP model is a right hand, so it cannot copy a left hand's chirality. With a left arm the robot does what your arm does in the same directions: its hand goes where yours goes, its palm faces where your palm faces and its fingers point where yours point; rotations and movements go the same way. Only the thumb is necessarily on the other side. Finger joints map anatomically (index to index, spreading toward the thumb). An earlier version mirrored the palm orientation but not the translation, so for example turning the left palm to the right turned the robot palm to the left, and a sideways move pushed the robot arm across its body into an awkward configuration that it then stayed near; that was the "weird pose and strange motion".

The arm IK is a joint-limit-aware damped least-squares solver warm-started from the previous solution.

**Hand mapping.** Index, middle and ring LEAP joints follow the matching human joints: MCP, PIP and DIP flexion relative to your open hand, each multiplied by a gain from your open/fist calibration so a full human fist gives LEAP MCP/PIP/DIP = 1.35/1.45/1.0 rad; splay relative to your open hand drives LEAP abduction (faded out as the finger bends). The LEAP thumb's structure differs from a human thumb (its zero pose is a flat "L"), so it is solved by IK: its tip follows the direction and extension of your thumb from its base, scaled from your open-thumb length to the LEAP thumb's reach. When your thumb tip comes within 0.6 palm widths of a fingertip, the target blends (fully at 0.3) toward the LEAP thumb and that LEAP fingertip meeting, and that finger may bend slightly to meet it. The earlier fingertip-position-only mapping ignored the fist calibration and placed the thumb relative to an open-hand pose the LEAP thumb cannot reach, which is why fingers and especially the thumb did not move as expected.

Limit handling follows `limit_policy` in the arm configuration:

- `constrain` (default): a wrist target outside the configured workspace (including the table-clearance floor) is projected onto its boundary, and the IK returns the closest joint-limited solution. The constrained, feasible command is executed and the state is `LIMIT_ACTIVE` with the reasons (workspace boundary, reach, joint limit) shown in both windows. This is the "constrain the infeasible portion" option of FR-27 / S-06.
- `reject`: the concept design's behaviour. An out-of-workspace target or an IK/thumb residual beyond `arm_fit_tolerance_m` / `hand_fit_tolerance_m` holds both parts of the pair (`LIMIT_ACTIVE`, no command).

With either policy the program keeps running. Joint commands are clamped into the model's joint ranges at every stage; a held pose read back from the simulation that sits fractionally past a limit (after contact or servo overshoot) is clamped instead of rejected. That rejection was the cause of the earlier "Command exceeds model joint limits" exit.

Residuals check the model solution, **not camera truth**. Every accepted joint command is clamped by a motion-rate bound, and its original source time determines expiry. The simulator independently rejects invalid, reordered-joint, future and expired signals. Its expired-signal behavior holds the simulated measured pose.

Optional UDP and JSONL output use the outer schema `quadradex.teleop.v1`. The same `control_signal` object is passed to the simulator. It is `null` while calibration/stop/loss/hold handling (or the reject policy) inhibits a pair. Packets also carry `fit_metrics` (IK residuals, `limit_active`, `limit_reasons`, `fit_warnings`) and `tracking` (per-landmark status, failure reasons, model and inference times). Within an accepted control signal:

| Field | Meaning |
| --- | --- |
| `schema` | `quadradex.control.v1` |
| `destination` | `simulation_only` |
| `joint_names`, `positions_rad` | Seven named arm joints, followed by LEAP motor IDs 0-15 |
| `arm_positions_rad` | Bounded demo-arm targets |
| `leap_motor_ids`, `leap_sim_positions_rad` | All 16 LEAP logical joint targets, explicitly in native motor-ID order |
| `leap_v1_sdk_positions_rad` | Logical radians + pi, following the official V1 SDK conversion |
| `source_received_monotonic_ns` | Original host frame receipt on `host_perf_counter_ns` |
| `generated_monotonic_ns`, `expires_monotonic_ns` | Generation and deadline on the same host clock |
| `velocity_limited` | Whether the target was reduced to obey configured motion rates |
| `hardware_enabled` | Always false in this prototype |

States: `STEREO_SETUP` (two-camera setup running, no command), `CALIBRATION_REQUIRED`, `CALIBRATING`, `READY`, `ALIGNING` (moving to the matched start pose at reduced speed), `STREAMING`, `LIMIT_ACTIVE`, `TRACKING_HOLD` (brief gap, no command, auto-continues), `TRACKING_LOST` (latched, needs Space), `CAMERA_ERROR` (no command, automatic retry), `FAULT` (recovered internal error, no command, needs Space), `STOPPED`. Packets also carry `start_pose` (live checks, robot reference match, aligning flag), `fault`, `camera_mode`, `cameras`, `stereo` (setup quality and per-joint fusion sources) and `tracking.partial_hold`.

The V1 SDK convention is a candidate interface for a future confirmed V1 hardware adapter. The values are not universal V2 commands, voltages, currents or force commands. Joint-ID order must not be inferred from MuJoCo/URDF traversal: joints 0 and 1, for example, traverse in the opposite order. Code and tests resolve every joint and actuator by name.

Transport is restricted to loopback. A future consumer must validate session/sequence, clock, source freshness, model revision, robot state and coordinated arm/hand limits before using a packet. No physical communication adapter is included because the LEAP hardware revision and arm/controller are not confirmed.

The simulation uses position servos, gravity compensation and model inertias. Robot links do not collide with each other, but the arm and hand collide with the bench and the movable objects (500 mL bottle, Rubik's cube, AirPods case, phone, with approximate dimensions), so contact and simple grasps can be previewed. The demo arm's masses, link geometry, servo gains and the convex-hull finger collision shapes are illustrative. It is not a collision planner or a calibrated predictor of loaded physical hardware. The visual motion under the exported targets is useful for retargeting review; it does not establish grip force, payload, contact stability or physical stopping performance.

## Accuracy: what is checked and what is still needed

The default completed-pair age limit is 120 ms, leaving nominal room inside the project's 150 ms sensor-to-command budget. It is a prototype choice. Host receipt is later than sensor exposure; end-to-end timing must include camera buffering, inference, filtering, IK, transport and the actual device command issuance. The requested output cadence is 25 Hz. Full-arm and finger inference may complete slower on a given computer; repeated or expired observations cannot count as fresh control updates.

**Physical arm accuracy has not been established.** Camera quality gates permit experimental simulation; packets retain `arm_accuracy = not_independently_validated`. Before physical integration, compare tracked 3D shoulder/elbow/wrist positions with an independent measurement system across the intended workspace, lighting, arm poses, motion speeds and occlusions. Align coordinate frames and timestamps. Also evaluate the actual follower wrist and fingertips under the project's separate retargeting procedures.

`reference_template.csv` provides the required columns. Collect at least ten matched samples of each point (`shoulder`, `elbow`, `wrist`) in a common calibrated coordinate frame. Do not use the same vision model as its own reference. Then run, using the launcher's Python environment:

```powershell
.\cv_hand_tracking\.venv\Scripts\python.exe -m teleoperation.evaluate_tracking measurements.csv --output teleoperation\reports\arm_validation.json --reference-description "Describe the independent measurement and frame alignment" --max-error-mm 60
```

The evaluator rejects non-finite data, missing point sets and timestamp mismatches over 20 ms by default. It reports mean/p95/maximum Euclidean error per point and passes only when every sampled point is within the configured maximum. The default 60 mm is a proposed component-level screening threshold inspired by PR-03, not an automatic allocation of the follower's complete error budget. Passing a dataset applies to those recorded conditions and never automatically enables actuation. Use a stricter threshold if sensing must consume only part of the 60 mm follower-error budget.

## Verification

Run both suites:

```powershell
.\cv_hand_tracking\.venv\Scripts\python.exe -m unittest discover -s cv_hand_tracking -v
.\cv_hand_tracking\.venv\Scripts\python.exe -m unittest teleoperation.test_teleoperation -v
```

- Original 17 hand tests pass; 50 arm/control/simulation tests pass. Besides the single-camera tracking, calibration, start-pose matching, hand-mapping, limit and never-exit tests, they cover: the left arm (start pose matched, motion and palm rotation in the same direction as the operator, fingers/pinch/spread like the right hand), tracking stability (hand window holds still and re-centres only near its edge, a lost hand's search window grows, fingertips slightly past the image edge are accepted, a hand near its previous position is kept when the pose wrist is off, segment-length stabilization), partial holds (a missing hand keeps the arm streaming and vice versa; both missing beyond the hold becomes a tracking gap), and the two-camera mode: calibration from the operator's body recovers a known camera pose (rotation within 1 degree, baseline within 6 %, reprojection under 3 px), save/reload/camera matching, fused forearm accurate to under 8 mm and finger joint angles to under 5 degrees with both views, recovery of an elbow/wrist/hand hidden in either camera (under 15 mm), left-arm fusion, time pairing, and a full two-camera session (setup, calibration, alignment, streaming while the wrist and hand are hidden in the side view).
- Hand-mapping and two-camera tests use synthetic operators (`synthetic_hand.py`, `synthetic_scene.py`: known 3D body/hand, two virtual cameras with different fields of view, pixel noise) driven through the real feature extraction, fusion and retargeting. They verify the algorithms, not MediaPipe's accuracy on real images.
- The rewritten camera worker was run on the real MediaPipe models with a public photo (both arms, about 15 ms pose + 19 ms hand), and the fusion was run on real per-view MediaPipe output to check the data path. Scripted end-to-end runs of the application with real UI rendering and the real simulator passed in both modes: one camera (calibration, alignment, a pinch, a limit excursion with guidance, a camera failure with automatic reconnection, exit only on Q) and two cameras (setup from a moving operator, calibration, alignment, streaming with the wrist and hand hidden from the side camera, no command in any inhibited state).
- **No live test with real cameras and a moving operator has been done for this revision**, in either mode. Please check single-camera stability, the two-camera setup and fusion, and the left arm with your cameras.

## Configure the demo arm

Edit a copy of `demo_arm.json` and pass `--arm-config`. It describes the seven-joint topology: shoulder yaw / pitch / upper-arm roll, elbow, forearm roll, wrist pitch / yaw, with the wrist centre as the IK point. Fields: upper-arm/forearm lengths, shoulder height (`base_height_m`) and bench height, home joint angles, joint limits, motion rates (`max_velocity_rad_s`, `hand_max_velocity_rad_s`), translation gain and `scale_by_arm_length`, hand mount offset, workspace bounds (the lower z bound is the table-clearance floor for the wrist centre), `limit_policy`, IK-fit tolerances, `match_start_pose`, `align_speed_scale` (fraction of the motion rates used while aligning) and `scene_objects`. The demo elbow allows up to 2.6 rad (149 degrees) so the raised-hand start pose is not at a limit. An arbitrary physical arm URDF requires a corresponding model adapter; changing the JSON alone does not make the scene that physical robot. The motion rates are tuned for responsive simulation and must be re-derived for real hardware.

Once your team chooses the arm and confirms the LEAP revision, replace the relevant model/adapter, set measured limits and mounting transforms, validate tracking and retargeting, and integrate the coordinated execution and independent stop architecture.

## Sources

- [Official LEAP simulation repository](https://github.com/leap-hand/LEAP_Hand_Sim), revision `150bc3d4b61fd6619193ba5a8ef209f3609ced89`; original URDF/STL files and MIT license are in `assets/leap_v1/`. `SOURCE.json` records file hashes.
- [Official LEAP V1 coordinate conversion](https://github.com/leap-hand/LEAP_Hand_API/blob/b0d00c881d0119077b2cab771a6de44f5aaec904/python/leap_hand_utils/leap_hand_utils.py).
- [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) and [Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).
- [MuJoCo Python API and interactive viewer](https://mujoco.readthedocs.io/en/3.14.0/python.html).
- Project requirements and concept design in `../docs/`.
