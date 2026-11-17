# QuadraDex-Software

Software for the QuadraDex dexterous hand teleoperation robotics capstone.

The current prototype implements camera-based hand acquisition with MediaPipe Hand Landmarker. It provides continuous finger and palm geometry for a downstream LEAP Hand retargeting subsystem.

## Run the hand-tracking prototype

On 64-bit Windows with Python 3.11 installed, open PowerShell in this repository and run:

```powershell
.\cv_hand_tracking\run_tracker.bat
```

The launcher installs the required libraries into an isolated local environment and downloads the model on first use. Select a camera by name, confirm its live preview, and choose **Use camera**.

- Press **O** to capture the open pose, then **C** to capture the closed pose. The display shows preparation, progress, sample counts, pause reasons, and completion.
- Press **Space** to start or stop the target stream; **S** stops it.
- Press **V** to choose another camera, **H** to change the hand selection, or **M** to reverse the camera's hand labels.
- Press **Q** or **Esc** to quit.

See the [hand-tracking guide](cv_hand_tracking/README.md) for setup, camera options, output format, UDP integration, timing measurements, and limitations.

## Repository layout

- [`cv_hand_tracking/`](cv_hand_tracking/): tracker, camera picker, guided calibration, geometry, example receiver, and tests.
- [`docs/QuadraDex_spec_v1.1.pdf`](docs/QuadraDex_spec_v1.1.pdf): project requirements.
- [Concept design](<docs/Concept_Design_Quadradex (1).pdf>): architecture and subsystem interfaces.

## Verification

After the launcher has installed the dependencies:

```powershell
.\cv_hand_tracking\.venv\Scripts\python.exe -m unittest discover -s cv_hand_tracking -v
```

The current suite contains 17 offline checks. Real-model integration checks also exercised loss/recovery, local UDP delivery, and camera switching with simulated input. The detailed guide distinguishes those checks from physical-hand and robot validation.

## Integration status

This prototype emits hand geometry and optional local UDP targets. It does not send motor commands. Robot-specific inverse kinematics, coordinated arm/hand command validation, and the independent stop path remain downstream integration work. Single-camera depth and occluded-finger estimates require experimental accuracy validation.

Camera logs, personal calibration profiles, virtual environments, downloaded models, and generated sample previews are excluded from version control.
