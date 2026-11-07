# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

"Brain" of an autonomous model car (Bosch Future Mobility–style): Python code that runs on an NVIDIA Jetson Orin Nano. It does lane following (OpenCV + PID) and traffic-sign reaction (YOLO + RealSense depth), and drives a low-level ESP32 over UART. The ESP32 firmware lives in a separate repo ([Bosch-Aura/embedded](https://github.com/Bosch-Aura/embedded)) and is not changed from here. Docs and many comments are in Spanish.

## Running

There is no build step, linter config or test suite. All code lives under `brain/`, and imports are relative to that folder: `dashboard_server.py` adds `brain/` to `sys.path`.

```bash
./start_dashboard.sh          # kills old instance, runs the Flask dashboard on 0.0.0.0:5000
./restart.sh                  # same, also force-frees port 5000

# Direct, from brain/:
python dashboard/dashboard_server.py --port-name /dev/ttyCH341USB0 [--lane-detection] [--sign-detection] [--debug]
# --auto-connect picks the serial port interactively; with no port, connect from the web UI

# Offline lane-detection tuning on a recorded video (from brain/):
python lane_detection/test_video.py <video> [--threshold 150]
python camera/test_realsense.py   # RealSense sanity check
```

Environment: `brain/requirements.txt` only lists Flask, flask-cors and pyserial. OpenCV, torch and ultralytics come from the Jetson's CUDA builds, so create the venv with `python3 -m venv venv --system-site-packages`. Don't add those packages to requirements.

## Architecture (v1)

- **`dashboard/dashboard_server.py` is the entry point and composition root.** It finds the camera (RealSense, or a webcam as fallback) and the serial port, builds `CommandSender`, `AutoPilotController` and `SignController`, and wires them together. It also serves the single-page UI (`dashboard.html`), the REST control routes (`/autopilot/*`, `/sign_detection/*`, `/uart/*`), the SSE telemetry streams (`/events`, `*/status/stream`) and MJPEG debug views (`/debug/*`).
- **Two parallel control threads share one `CommandSender`:**
  - `lane_detection/AutoPilotController` (~30 Hz) runs this pipeline: `LaneDetector` (bird-view and sliding windows, in `lane_detector.py`) → `FilterController` (moving average) → `PIDController` → `AngleConverter` (degrees to servo value 50–160, with center at 105) → UART. It only sends a value when it changes. It also feeds lane-based intersection detections to `IntersectionEventController`, which debounces them and fires `intersection_strategies/`.
  - `sign_vision/SignController` (~15–20 Hz, limited by YOLO) gets detections from `SignDetector`. Each detection's YOLO class label is looked up in `self.strategies`, so **the dict keys must match the model's class names** (`stop`, `priority-road`, `park`, …).
- **Strategies take over the car by pausing the autopilot.** A `SignStrategy` (in `sign_vision/strategies/`, base class in `base_strategy.py`) checks `min_confidence` and `activation_distance`. It then calls `autopilot.pause()`, drives timed maneuvers straight through `CommandSender`, and calls `resume()`, which resets the PID and filter. While paused, the autopilot loop keeps running for visualization but sends nothing. This sign→autopilot coupling is known tech debt. `docs/v2.0.0/REFACTOR_PROPOSAL.md` proposes replacing it with a central `VehicleController` and an explicit state machine.
- **Model weights:** `SignDetector` prefers `sign_vision/weights/best.engine`, a TensorRT file built from `best.pt` when a GPU is present. `.engine` and `.onnx` files are gitignored and specific to each machine.

## ESP32 link (`brain/command_sender.py`)

- Protocol: `CHANNEL:COMMAND:VALUE\n` at 115200 baud (spec in `embedded/docs/BRAIN_TEAM_PROTOCOL.md`). On the Jetson the port is `/dev/ttyCH341USB0`; on a PC it is usually `/dev/ttyUSB0`.
- Speed: the UI and strategies use 0–255, which is mapped to duty 140–255 (the motor stalls below 140). Direction is sent with `C:SET_DIR` only when it changes.
- Steering: the raw servo value goes out as `C:SET_STEER`. If the car steers the wrong way, flip `STEER_INVERTED`.
- Lifecycle: opening the port resets the ESP32. The dashboard arms the board with `M:SYS_ARM` after `EVENT:SYSTEM_READY` arrives, or after 3 s if it doesn't. It sends a heartbeat (`M:PING`) every 50 ms, because the ESP32 goes to `FAULT` after 300 ms with no traffic. `FAULT` does not clear by itself; re-arm from the UI.
- Speed is latched on the ESP32: the brain sends it once and again only when it changes. `CommandSender` mirrors the latched duty (corrected by ESP32 events) and skips repeats; a stop (0) always goes out.
- The ESP32 owns the ultrasonic sensor and the resume: on an obstacle it keeps the speed (`EVENT:OBSTACLE_HOLD:<pwm>`) and resumes it by itself (`EVENT:OBSTACLE_RESUMED:<pwm>`). The Jetson re-sends nothing. `M:OBSTACLE_RESET` drops the kept speed. AUTO/MANUAL mode exists only on the Jetson side.

## Docs

`README.md` (software architecture document with sequence diagrams), `docs/ARCHITECTURE.md`, `docs/CLASS_DIAGRAM.md`, `brain/dashboard/README.md` and `brain/sign_vision/README.md` (strategy thresholds and behaviors).
