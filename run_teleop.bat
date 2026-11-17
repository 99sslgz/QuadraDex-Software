@echo off
setlocal
cd /d "%~dp0"
if not exist "cv_hand_tracking\.venv\Scripts\python.exe" (
    echo Creating the project Python environment...
    python -m venv cv_hand_tracking\.venv
    if errorlevel 1 exit /b 1
)
"cv_hand_tracking\.venv\Scripts\python.exe" -c "import mediapipe, cv2, cv2_enumerate_cameras, mujoco" >nul 2>&1
if errorlevel 1 (
    "cv_hand_tracking\.venv\Scripts\python.exe" -m pip install -r teleoperation\requirements.txt
    if errorlevel 1 exit /b 1
)
"cv_hand_tracking\.venv\Scripts\python.exe" -m teleoperation.run %*
if errorlevel 1 pause
