# Full-arm tracking, LEAP signals and simulation

This experimental branch adds synchronized camera tracking of a selected shoulder, elbow, wrist and all finger landmarks. A configurable seven-joint demo robot arm carries the official **right LEAP Hand V1** model. The simulation follows generated position-control signals.

The physical arm has not been selected and the physical LEAP revision is unconfirmed. The demo arm is an adjustable model, not a representation of purchased hardware. The V1 model follows the project's original LEAP baseline. This application generates V1-compatible signal values and drives the simulator; it does not connect to a motor bus or move physical actuators.

## Run

From the repository root, with 64-bit Python 3.11 installed:

```powershell
.\run_teleop.bat
```

The launcher installs dependencies into `cv_hand_tracking/.venv`. The body and hand inference models download once and stay outside Git. The LEAP meshes, URDF, MIT license and pinned source manifest are included in the repository.

Two windows open: a camera/calibration window and an interactive MuJoCo robot simulation. Select a camera in the named camera picker. The simulation initially holds its home pose.

1. Mount the camera far enough back to see **both shoulders, both hips, and the selected arm from shoulder to fingertips**. Keep one person in view. The default request is 1280 x 720 at 30 camera FPS. Camera resolution and a clear view matter: the hand occupies fewer pixels than in the original close-up hand prototype.
2. Select Right or Left with **H** / the **Hand** button. Right is the default. If the driver reverses left/right labels, press **M**. The camera view is mirrored only for display.
3. Hold a comfortable neutral arm pose with an open hand. Keep the elbow slightly bent and the palm visible. Press **O** / **1. Open**. The countdown and progress bar cover the simultaneous **open hand and arm neutral calibration**. Keep your arm still until it reports success.
4. Keep the arm visible, bend your fingers and thumb into a fist, and press **C** / **2. Closed**. Wait for calibration to finish. Brief missing measurements pause the capture; movement restarts the stable sampling window. A timeout or insufficient finger travel gives an explicit retry instruction.
5. With good tracking for at least half a second, press **Space**. Move slowly at first. The simulated arm follows wrist displacement and palm rotation relative to the neutral pose; redundant arm motion also follows the elbow where the model permits. The four LEAP fingers follow geometric fingertip/direction targets. Pinky landmarks remain diagnostic.
6. **S** stops the paired arm/hand stream. **Space** stops or deliberately resumes it. **R** clears calibration. **V** / **Camera** opens camera selection again. Camera/side changes inhibit targets and require new calibration. **Q/Esc** closes the tracking application. Closing the simulator also ends the session.

In the 3D window, drag to orbit and scroll to zoom. Press **F** with that window focused to follow a close view of the hand; press F again for the arm view. Tracking shortcuts belong to the camera window.

Arm/hand calibration is session-local. It is deliberately not loaded from the original hand-only calibration file: an arm anchor and a camera/side association must be established together.

## Try the simulator without a camera

```powershell
.\run_teleop.bat --demo
```

This produces clearly labeled **synthetic demo** arm/finger signals. It demonstrates simulator motion and signal conventions, not camera accuracy. A finite run with numerical logs and a robot-only image:

```powershell
.\run_teleop.bat --demo --headless --duration 8 --log teleoperation\logs\demo_001.jsonl --snapshot teleoperation\reports\demo.png
```

Use a new log filename per run. Camera images/video are not saved by the logger; `--snapshot` renders the simulated robot only.

Other options:

```powershell
.\run_teleop.bat --list-cameras
.\run_teleop.bat --camera 1 --side right
.\run_teleop.bat --camera 1 --backend dshow
.\run_teleop.bat --download-models
.\run_teleop.bat --arm-config teleoperation\demo_arm.json
.\run_teleop.bat --udp 127.0.0.1:5006 --log teleoperation\logs\trial_001.jsonl
```

The original `cv_hand_tracking/run_tracker.bat` remains a separate hand-only tool.

## What the new system measures

MediaPipe Pose Landmarker Lite provides the torso, shoulder, elbow and wrist estimates. The selected wrist defines an image crop for Hand Landmarker, giving fingers more image detail. **Both models process the same captured frame and preserve the same host-receipt timestamp.** The crop's image coordinates are mapped back into the full camera image. A wrist proximity check rejects a hand that does not correspond to the selected arm.

The inference worker reads only the latest frame, and the UI reads only the latest completed pair. There is no backlog of old poses. Raw body geometry, visibility/presence, crop coverage, plausible bone lengths, temporal wrist continuity and hand geometry are checked before paired control becomes available. Filters reduce jitter; they do not fill in an occluded arm.

The body coordinate frame has its origin at the selected shoulder, x forward, y toward the operator's left and z up. Both shoulders and hips define the torso axes. This makes retargeting relative to the torso instead of treating the model's hip-centred coordinates as calibrated global camera depth. Elbow flexion, upper-arm/forearm length estimates, and the shoulder/elbow/wrist points are available in logs.

Visibility is an estimate of observability, **not proof of metric accuracy**. Single-view inferred depth, self-occlusion, sleeve/object occlusion and wrist axial rotation remain uncertain. Absolute camera-frame translation and full anatomical shoulder/forearm twist are not uniquely recovered by three arm points. The palm provides the wrist orientation estimate. This CV extension does not replace the specification's encoder-equipped leader exoskeleton for formal verification without an agreed requirement change and validation evidence.

## Mapping and control signals

`robot_model.py` builds a MuJoCo scene from the configurable demo arm and the pinned LEAP URDF. Published LEAP joint transforms, limits, mesh placements and inertias are used directly. Fingertip sites are reproducible distal-surface estimates computed from the published STL meshes; they are not measured contact pads.

The arm IK targets calibrated wrist displacement and palm orientation, with the observed elbow used as a secondary objective in the arm's redundant degree of freedom. Hand IK fits fingertip positions and distal directions relative to the open calibration, scaled for the robot's palm width. Left-hand input is reflected into the right V1 hand's geometric convention. Human joint angles are not blindly copied into LEAP motor slots.

Position, orientation and mean fingertip fit residuals must pass their configured bounds before a pair is dispatched. These residuals check the model solution, **not camera truth**. Out-of-workspace or poorly matched targets hold both parts of the pair. Stale or missing sensing latches tracking loss and requires explicit resume. Every accepted joint command is clamped by a motion-rate bound, and its original source time determines expiry. The simulator independently rejects invalid, reordered-joint, future and expired signals. Its expired-signal behavior holds the simulated measured pose.

Optional UDP and JSONL output use the outer schema `quadradex.teleop.v1`. The same `control_signal` object is passed to the simulator. It is `null` while calibration/stop/loss/limit handling inhibits a pair. Within an accepted control signal:

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

The V1 SDK convention is a candidate interface for a future confirmed V1 hardware adapter. The values are not universal V2 commands, voltages, currents or force commands. Joint-ID order must not be inferred from MuJoCo/URDF traversal: joints 0 and 1, for example, traverse in the opposite order. Code and tests resolve every joint and actuator by name.

Transport is restricted to loopback. A future consumer must validate session/sequence, clock, source freshness, model revision, robot state and coordinated arm/hand limits before using a packet. No physical communication adapter is included because the LEAP hardware revision and arm/controller are not confirmed.

The simulation uses position servos, gravity compensation and model inertias. The demo arm's masses, link geometry and servo gains are illustrative. Mesh contacts/self-collision are disabled in this motion-preview scene. It is not a collision planner or a calibrated predictor of loaded physical hardware. The visual motion under the exported targets is useful for retargeting review; it does not establish grip force, payload, contact stability or physical stopping performance.

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

- Original 17 tests pass; 14 additional tests cover torso-frame invariance, visibility/cropping/length rejection, continuity, neutral calibration, the official 23-joint scene and explicit motor order, arm/hand FK-to-IK round trips, paired limit rejection, V1 conversion, speed limits, source expiry, simulator signal consumption and independent-reference evaluation.
- Known-model round-trip checks reached sub-millimetre wrist and fingertip residuals in the tested poses. These verify solver/model consistency, not observed human tracking accuracy.
- A synthetic end-to-end application test exercised O/C calibration, start, IK, generated packets, simulated motion, a tracking interruption and deliberate resume. It confirmed that inhibited states exported no control signal.
- A real-model test on Google's public pose image returned body estimates and some valid synchronized hand estimates; the original source frame was preserved through both models. This test is limited by the small hand in that photo and does not establish robust hand tracking across poses.
- A short live camera diagnostic opened the camera and ran both models. It rejected the incomplete torso view rather than generating control. Physical full-arm range-of-motion and accuracy tests remain operator work.
- Robot-only rendered snapshots were visually inspected. The synthetic demo exercises generated arm/finger positions through the simulated position servos.

## Configure the demo arm

Edit a copy of `demo_arm.json` and pass `--arm-config` to adjust upper-arm/forearm lengths, base height, home joint angles, joint limits, motion rates, palm mount, workspace bounds, translation gain and IK-fit limits. This configuration describes the existing seven-joint topology (base yaw, shoulder pitch/roll, elbow, wrist yaw/pitch/roll). An arbitrary physical arm URDF requires a corresponding model adapter; changing the JSON name alone does not make the scene that physical robot.

Once your team chooses the arm and confirms the LEAP revision, replace the relevant model/adapter, set measured limits and mounting transforms, validate tracking and retargeting, and integrate the coordinated execution and independent stop architecture.

## Sources

- [Official LEAP simulation repository](https://github.com/leap-hand/LEAP_Hand_Sim), revision `150bc3d4b61fd6619193ba5a8ef209f3609ced89`; original URDF/STL files and MIT license are in `assets/leap_v1/`. `SOURCE.json` records file hashes.
- [Official LEAP V1 coordinate conversion](https://github.com/leap-hand/LEAP_Hand_API/blob/b0d00c881d0119077b2cab771a6de44f5aaec904/python/leap_hand_utils/leap_hand_utils.py).
- [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker) and [Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).
- [MuJoCo Python API and interactive viewer](https://mujoco.readthedocs.io/en/3.14.0/python.html).
- Project requirements and concept design in `../docs/`.
