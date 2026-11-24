#!/usr/bin/env python3
"""
Flask server for the car control dashboard (Nucleo/STM32 over UART).
Serves the dashboard and proxies commands to the Nucleo via serial.
"""

from flask import Flask, send_from_directory, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import sys
import threading
import time
import queue
import json

# Import auto-pilot modules
import sys
import os
# Add parent directory to path to import brain modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Store import errors for debugging
import_errors = {}

try:
    from camera.video_streamer import VideoStreamer
except ImportError as e:
    print(f"Warning: Could not import VideoStreamer: {e}")
    VideoStreamer = None
    import_errors['VideoStreamer'] = str(e)

try:
    from camera.realsense_streamer import RealSenseStreamer
except ImportError as e:
    print(f"Warning: Could not import RealSenseStreamer: {e}")
    RealSenseStreamer = None
    import_errors['RealSenseStreamer'] = str(e)

try:
    from lane_detection.autopilot_controller import AutoPilotController
except ImportError as e:
    print(f"Warning: Could not import AutoPilotController: {e}")
    AutoPilotController = None
    import_errors['AutoPilotController'] = str(e)
except Exception as e:
    # Catch other exceptions (syntax errors, etc.)
    print(f"Warning: Error importing AutoPilotController: {e}")
    AutoPilotController = None
    import_errors['AutoPilotController'] = str(e)

try:
    from command_sender import CommandSender
    from command_sender import (
        set_protocol as set_uart_protocol,
        get_protocol as get_uart_protocol,
        PROTOCOL_ESP32,
        SUPPORTED_PROTOCOLS,
    )
except ImportError as e:
    print(f"Warning: Could not import CommandSender: {e}")
    CommandSender = None
    import_errors['CommandSender'] = str(e)

try:
    from lane_detection.lane_detector import MarcosLaneDetector_Advanced
except ImportError as e:
    print(f"Warning: Could not import MarcosLaneDetector_Advanced: {e}")
    MarcosLaneDetector_Advanced = None
    import_errors['MarcosLaneDetector_Advanced'] = str(e)

# Check if any critical modules are missing
if VideoStreamer is None or AutoPilotController is None or CommandSender is None or MarcosLaneDetector_Advanced is None:
    print("Warning: Some autopilot modules are not available. Auto-pilot features will be disabled.")

# Import sign vision modules
try:
    from sign_vision.sign_detector import SignDetector
except ImportError as e:
    print(f"Warning: Could not import SignDetector: {e}")
    SignDetector = None
    import_errors['SignDetector'] = str(e)
except Exception as e:
    # Catch other exceptions (syntax errors, etc.)
    print(f"Warning: Error importing SignDetector: {e}")
    SignDetector = None
    import_errors['SignDetector'] = str(e)

try:
    from sign_vision.sign_controller import SignController
except ImportError as e:
    print(f"Warning: Could not import SignController: {e}")
    SignController = None
    import_errors['SignController'] = str(e)

try:
    from safety.ultra_uart import ThreadUltraUART
except ImportError as e:
    print(f"Warning: Could not import ThreadUltraUART: {e}")
    ThreadUltraUART = None
    import_errors['ThreadUltraUART'] = str(e)
except Exception as e:
    print(f"Warning: Error importing ThreadUltraUART: {e}")
    ThreadUltraUART = None
    import_errors['ThreadUltraUART'] = str(e)

# Try to import serial
try:
    import serial
    from serial.tools import list_ports as serial_list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("Warning: pyserial not installed. Install with: pip install pyserial", file=sys.stderr)

# Check CUDA availability
CUDA_AVAILABLE = False
CUDA_DEVICE_COUNT = 0
CUDA_ERROR = None
try:
    import cv2
    cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
    CUDA_AVAILABLE = True
    CUDA_DEVICE_COUNT = cuda_device_count
except Exception as e:
    CUDA_AVAILABLE = False
    CUDA_DEVICE_COUNT = 0
    CUDA_ERROR = str(e)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# UART Configuration
UART_BAUD_RATE = 115200
UART_TERMINATOR = ";;\r\n"

# Global serial connection
serial_conn = None
serial_lock = threading.Lock()
serial_message_queue = queue.Queue()
serial_reader_thread = None
serial_reader_running = False
serial_read_buffer = ""  # Buffer for incomplete serial lines

# Heartbeat sender thread
heartbeat_sender_thread = None
heartbeat_sender_running = False

# Global event clients (combined stream)
event_clients = []
event_clients_lock = threading.Lock()

# SSE clients for separate streams (maintained for backward compatibility if needed, but we will consolidate)
# Note: We will route ALL events to event_clients as well

# SSE clients for autopilot status streaming (Deprecated - mapped to event_clients)
autopilot_status_broadcast_running = False
autopilot_status_broadcast_thread = None

# SSE clients for sign detection status streaming (Deprecated - mapped to event_clients)
sign_detection_status_broadcast_running = False
sign_detection_status_broadcast_thread = None

# ...

# System state tracking
# Default state: armed, manual mode (Nucleo)
system_state = {'mode': 'MANUAL', 'state': 'RUNNING'}
system_state_lock = threading.Lock()

# Auto-pilot components
video_streamer = None
autopilot_controller = None
command_sender = None

# Sign vision components
sign_detector = None
sign_detection_controller = None
sign_detection_initializing = False  # True while loading YOLO model in background

# UltraSafety components (ESP32 ultrasonic reader + safety supervisor)
ultra_thread = None
ultra_safety_thread = None
ultra_safety_running = False
ultra_safety_enabled = True
ultra_last_ui_speed = 0
ultra_last_status = {
    "ultra_blocking": False,
    "saved_speed_ui": None,
    "stale": True,
    "valid": False,
    "present": False,
    "dist_cm": None,
    "last_update_age_ms": None,
}
ultra_status_lock = threading.Lock()

app = Flask(__name__, static_folder=SCRIPT_DIR)
CORS(app)  # Enable CORS for all routes


def write_uart_command(command: str):
    """Write a UART command, with the line terminator the active protocol expects.

    - nucleo: keep ';;\\r\\n' intact (do not strip or the Nucleo may not parse).
    - esp32: terminate with a single '\\n'. The ESP32 splits lines on '\\r' OR '\\n',
      so '\\r\\n' would make it receive an extra empty line and log a parse error.
    """
    global serial_conn
    if not serial_conn or not serial_conn.is_open:
        return False, "Serial port not connected"

    try:
        with serial_lock:
            if get_uart_protocol() == PROTOCOL_ESP32:
                command = command.rstrip("\r\n") + "\n"
            elif not command.endswith("\n"):
                command = command + "\r\n"
            line = command.encode("utf-8")
            serial_conn.write(line)
            serial_conn.flush()
            print(f"[UART TX] {line.decode('utf-8', errors='replace').strip()!r}")
            return True, "OK"
    except Exception as e:
        return False, str(e)


def translate_http_to_uart(endpoint: str, args: dict):
    """Translate HTTP endpoint to a UART command for the active protocol (see command_sender)."""
    from command_sender import (
        SERVO_CENTER, SERVO_RIGHT, SERVO_LEFT, SPEED_UI_MAX,
        build_arm, build_disarm, build_mode, build_brake,
        build_speed, build_steer, build_lights,
    )
    FORWARD_SPEED = 220
    no_reverse_msg = "Backward not supported by the ESP32 firmware (no reverse)"

    if endpoint == 'arm':
        return True, "ARM", build_arm()
    elif endpoint == 'disarm':
        return True, "DISARM", build_disarm()
    elif endpoint == 'mode':
        value = args.get('value', '').upper()
        if value == 'AUTO':
            return True, "MODE: AUTO", build_mode(True)
        elif value == 'MANUAL':
            return True, "MODE: MANUAL", build_mode(False)
        else:
            return False, "Invalid mode value", None
    elif endpoint == 'brake':
        return True, "BRAKE", build_brake()
    elif endpoint == 'forward':
        return True, "FORWARD", build_speed(FORWARD_SPEED, "forward")
    elif endpoint == 'back':
        cmd = build_speed(SPEED_UI_MAX, "backward")
        if cmd is None:
            return False, no_reverse_msg, None
        return True, "BACKWARD", cmd
    elif endpoint == 'driveStop':
        return True, "STOP", build_speed(0)
    elif endpoint == 'changeSpeed':
        speed_str = args.get('speed', '0')
        direction_str = args.get('direction', 'forward')
        try:
            speed = int(speed_str)
            if speed < 0 or speed > SPEED_UI_MAX:
                return False, f"Speed must be 0-{SPEED_UI_MAX}", None
            if speed == 0:
                return True, "STOP", build_speed(0)
            cmd = build_speed(speed, direction_str)
            if cmd is None:
                return False, no_reverse_msg, None
            return True, f"SPEED: {speed} ({direction_str})", cmd
        except ValueError:
            return False, "Invalid speed value", None
    elif endpoint == 'steer' or endpoint == 'changeSteer':
        angle_str = args.get('angle', str(SERVO_CENTER))
        try:
            angle = int(angle_str)
            if angle < SERVO_RIGHT or angle > SERVO_LEFT:
                return False, f"Angle must be {SERVO_RIGHT}-{SERVO_LEFT}", None
            return True, f"STEER: {angle}°", build_steer(angle)
        except ValueError:
            return False, "Invalid angle value", None
    elif endpoint == 'left':
        return True, "LEFT", build_steer(SERVO_LEFT)
    elif endpoint == 'right':
        return True, "RIGHT", build_steer(SERVO_RIGHT)
    elif endpoint == 'steerStop':
        return True, "CENTER", build_steer(SERVO_CENTER)
    elif endpoint in ('LightsOn', 'LightsOff', 'LightsAuto'):
        state = endpoint[len('Lights'):]
        cmd = build_lights(state)
        if cmd is None:
            return False, "Lights not supported", None
        return True, f"LIGHTS {state.upper()}", cmd
    else:
        return False, f"Unknown endpoint: {endpoint}", None


@app.route('/')
def index():
    """Serve the dashboard HTML file."""
    return send_from_directory(SCRIPT_DIR, 'dashboard.html')


@app.route('/dashboard.css')
def serve_css():
    """Serve the dashboard CSS file."""
    return send_from_directory(SCRIPT_DIR, 'dashboard.css')


@app.route('/fonts/<path:filename>')
def serve_fonts(filename):
    """Serve font files."""
    import os
    fonts_dir = os.path.join(SCRIPT_DIR, 'fonts')
    return send_from_directory(fonts_dir, filename)


@app.route('/<path:endpoint>')
def handle_command(endpoint):
    """Handle all dashboard commands and translate to UART."""
    args = request.args.to_dict()

    success, message, uart_cmd = translate_http_to_uart(endpoint, args)

    if not success:
        return jsonify({'error': message}), 400

    if uart_cmd is None:
        return jsonify({'status': 'ok', 'message': message}), 200

    cmd_success, cmd_message = write_uart_command(uart_cmd)

    if cmd_success:
        # Mantener current_speed del SignController sincronizado con la velocidad
        # que el dashboard manda, para que last_speed_before_stop sea siempre correcto.
        if sign_detection_controller is not None:
            if endpoint == 'forward':
                sign_detection_controller.update_current_speed(220)
            elif endpoint == 'changeSpeed':
                try:
                    speed_val = int(args.get('speed', 0))
                    direction = args.get('direction', 'forward')
                    if speed_val > 0 and direction.lower() == 'forward':
                        sign_detection_controller.update_current_speed(speed_val)
                except (ValueError, TypeError):
                    pass
            # driveStop y brake no actualizan current_speed: queremos conservar
            # last_speed_before_stop como referencia para el resume.

        return jsonify({'status': 'ok', 'message': message, 'uart_command': uart_cmd}), 200
    else:
        return jsonify({'error': f'Failed to send UART command: {cmd_message}'}), 500


@app.route('/proxy/<path:endpoint>')
def proxy_request(endpoint):
    """
    Proxy requests to external device (legacy) to avoid CORS issues.
    Usage: /proxy/arm, /proxy/status, etc.
    """
    esp32_ip = request.args.get('ip', '192.168.4.1')
    url = f'http://{esp32_ip}/{endpoint}'

    try:
        import requests
        response = requests.get(url, timeout=2)
        return response.text, response.status_code
    except ImportError:
        return jsonify({'error': 'requests library not installed. Install with: pip install requests'}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def open_serial(port: str, baud: int = UART_BAUD_RATE):
    """Open serial port with appropriate settings."""
    if not SERIAL_AVAILABLE:
        raise RuntimeError("pyserial not installed")

    ser = serial.Serial()
    ser.port = port
    ser.baudrate = baud
    ser.timeout = 0.1  # Shorter timeout for more responsive reading
    ser.write_timeout = 1.0
    ser.bytesize = serial.EIGHTBITS
    ser.parity = serial.PARITY_NONE
    ser.stopbits = serial.STOPBITS_ONE
    ser.xonxoff = False
    ser.rtscts = False
    ser.dsrdtr = False
    # Keep DTR/RTS low BEFORE opening: on the ESP32 DevKit (CH340 auto-reset
    # circuit) asserting them on open resets the board (~10 s boot delay) or
    # holds it in reset. Harmless for the Nucleo ST-Link.
    ser.dtr = False
    ser.rts = False
    ser.open()
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def read_available_lines(ser):
    """Read all available lines from serial port, buffering incomplete lines."""
    global serial_read_buffer
    lines = []
    if not ser or not ser.is_open:
        return lines

    try:
        # Read all available data first
        available = ser.in_waiting
        if available > 0:
            # Read in chunks to avoid blocking
            data = ser.read(available)
            # Decode and append to buffer
            new_data = data.decode('utf-8', errors='ignore')
            serial_read_buffer += new_data
            
            # Process complete lines (ending with \n or \r\n)
            # Keep processing until no more complete lines in buffer
            while True:
                # Check for \r\n first (Windows line ending)
                if '\r\n' in serial_read_buffer:
                    line, serial_read_buffer = serial_read_buffer.split('\r\n', 1)
                    line = line.strip()
                    if line:
                        # Fix for concatenated messages (e.g. EVENT:A...EVENT:B...)
                        # If a line contains multiple "EVENT:" markers after the start, split them
                        if 'EVENT:' in line[6:]:  # Skip checking the very start
                            parts = line.split('EVENT:')
                            # First part (might be empty if line started with EVENT:)
                            first = parts[0].strip()
                            if first:
                                lines.append(first)
                            
                            # Subsequent parts need "EVENT:" added back
                            for part in parts[1:]:
                                clean_part = f"EVENT:{part}".strip()
                                if clean_part:
                                    lines.append(clean_part)
                        else:
                             lines.append(line)

                # Then check for \n (Unix line ending)
                elif '\n' in serial_read_buffer:
                    line, serial_read_buffer = serial_read_buffer.split('\n', 1)
                    line = line.strip()
                    if line:
                        # Fix for concatenated messages (duplicate logic)
                        if 'EVENT:' in line[6:]:
                            parts = line.split('EVENT:')
                            first = parts[0].strip()
                            if first:
                                lines.append(first)
                            for part in parts[1:]:
                                clean_part = f"EVENT:{part}".strip()
                                if clean_part:
                                    lines.append(clean_part)
                        else:
                            lines.append(line)
                else:
                    # No more complete lines, break
                    break
        else:
            # If no data available, try readline with short timeout
            # This helps catch any remaining complete lines
            try:
                data = ser.readline()
                if data:
                    line = data.decode('utf-8', errors='ignore').rstrip()
                    if line:
                        # Fix for concatenated messages in readline fallback
                        if 'EVENT:' in line[6:]:
                            parts = line.split('EVENT:')
                            first = parts[0].strip()
                            if first:
                                lines.append(first)
                            for part in parts[1:]:
                                clean_part = f"EVENT:{part}".strip()
                                if clean_part:
                                    lines.append(clean_part)
                        else:
                            lines.append(line)
            except:
                pass  # Timeout is expected when no data
    except Exception as e:
        print(f"[DEBUG] Read error: {e}", file=sys.stderr)
        pass

    return lines


def parse_system_events(line: str):
    """Parse system state and mode events from serial messages."""
    global system_state

    if line.startswith('EVENT:STATE_CHANGED:'):
        # Extract only the first word to avoid trailing garbage/extra messages
        raw_state = line.replace('EVENT:STATE_CHANGED:', '').strip()
        state = raw_state.split()[0] if raw_state else raw_state
        
        with system_state_lock:
            system_state['state'] = state
            print(f"[SystemState] State changed to: {state} (raw: {raw_state})")
    elif line.startswith('EVENT:MODE_CHANGED:'):
        # Extract only the first word
        raw_mode = line.replace('EVENT:MODE_CHANGED:', '').strip()
        mode = raw_mode.split()[0] if raw_mode else raw_mode
        
        with system_state_lock:
            system_state['mode'] = mode
            print(f"[SystemState] Mode changed to: {mode} (raw: {raw_mode})")
    elif line.startswith('@kl:'):
        raw_value = line.replace('@kl:', '').replace(';;', '').strip()
        try:
            kl_value = int(raw_value)
        except ValueError:
            kl_value = None

        if kl_value == 0:
            state = "DISARMED"
            mode = "MANUAL"
        elif kl_value == 15:
            state = "ARMED"
            mode = "MANUAL"
        elif kl_value == 30:
            state = "RUNNING"
            mode = "AUTO"
        else:
            state = "UNKNOWN"
            mode = "MANUAL"

        with system_state_lock:
            system_state['state'] = state
            system_state['mode'] = mode
            print(f"[SystemState] KL update: {kl_value} -> {state}/{mode}")
    elif line.startswith('@shutdown:'):
        with system_state_lock:
            system_state['state'] = "SHUTDOWN"
            print("[SystemState] Shutdown reported by STM32")


def serial_reader_worker():
    """Background thread that reads from serial port."""
    global serial_conn, serial_reader_running, event_clients, sign_detection_controller, ultra_last_ui_speed

    while serial_reader_running:
        if serial_conn and serial_conn.is_open:
            try:
                lines = read_available_lines(serial_conn)
                for line in lines:
                    if line:
                        # Parse system events for state tracking
                        parse_system_events(line)
                        if line.startswith('@speed:'):
                            raw_speed = line.replace('@speed:', '').replace(';;', '').strip()
                            try:
                                speed_mm_s = int(raw_speed)
                                ui_speed = int(round((abs(speed_mm_s) / 500) * 255))
                                ultra_last_ui_speed = ui_speed
                                if sign_detection_controller is not None:
                                    sign_detection_controller.update_current_speed(ui_speed)
                            except ValueError:
                                pass

                        # Broadcast message to all connected SSE clients
                        message_data = {
                            'type': 'serial',
                            'message': line,
                            'timestamp': time.time()
                        }
                        # Send to unified event clients
                        with event_clients_lock:
                            # Send to all connected clients
                            # Copy list to avoid modification during iteration
                            for client_queue in event_clients[:]:
                                try:
                                    client_queue.put(message_data, block=False)
                                except queue.Full:
                                    # Client queue is full, remove it (client might be disconnected)
                                    try:
                                        event_clients.remove(client_queue)
                                    except ValueError:
                                        pass  # Already removed
                if not lines:
                    time.sleep(0.01)
            except Exception as e:
                if serial_reader_running:
                    print(f"[SerialReader] Error: {e}", file=sys.stderr)
                    time.sleep(0.1)
        else:
            time.sleep(0.1)


def start_serial_reader():
    """Start the serial reader thread."""
    global serial_reader_thread, serial_reader_running

    if serial_reader_thread is None or not serial_reader_thread.is_alive():
        serial_reader_running = True
        serial_reader_thread = threading.Thread(
            target=serial_reader_worker, daemon=True)
        serial_reader_thread.start()


def stop_serial_reader():
    """Stop the serial reader thread."""
    global serial_reader_running, serial_reader_thread

    serial_reader_running = False
    if serial_reader_thread:
        serial_reader_thread.join(timeout=1.0)
        serial_reader_thread = None


def heartbeat_worker():
    """Send heartbeat (#alive / M:HEARTBEAT) every 50ms so the controller watchdog (~120ms) does not stop the motors."""
    global serial_conn, heartbeat_sender_running, command_sender

    while heartbeat_sender_running:
        if serial_conn and serial_conn.is_open and command_sender:
            try:
                command_sender.send_heartbeat()
            except Exception as e:
                print(f"[Heartbeat] Error: {e}", file=sys.stderr)
        time.sleep(0.05)  # 50ms; watchdog is 120ms on both ESP32 and Nucleo

    print("[Heartbeat] Worker thread stopped")


def start_heartbeat_sender():
    """Start the heartbeat sender thread."""
    global heartbeat_sender_thread, heartbeat_sender_running, command_sender

    # When connecting from the CLI (--port-name / --auto-connect) nothing else
    # creates the CommandSender, and without it no heartbeat is ever sent.
    if command_sender is None and CommandSender is not None:
        command_sender = CommandSender(write_uart_command)

    if heartbeat_sender_thread is None or not heartbeat_sender_thread.is_alive():
        heartbeat_sender_running = True
        heartbeat_sender_thread = threading.Thread(
            target=heartbeat_worker, daemon=True)
        heartbeat_sender_thread.start()


def stop_heartbeat_sender():
    """Stop the heartbeat sender thread."""
    global heartbeat_sender_running, heartbeat_sender_thread

    heartbeat_sender_running = False
    if heartbeat_sender_thread:
        heartbeat_sender_thread.join(timeout=1.0)
        heartbeat_sender_thread = None


ULTRA_UART_PORT = "/dev/ttyTHS1"
ULTRA_UART_BAUD = 115200
ULTRA_STALE_SECONDS = 0.25
ULTRA_CHECK_INTERVAL = 0.05
ULTRA_CLEAR_DELAY = 0.2
ULTRA_MIN_STOP_INTERVAL = 0.2


def start_ultra_thread():
    """Start ultrasonic UART reader thread (ESP32 -> Jetson)."""
    global ultra_thread

    if ThreadUltraUART is None:
        print("[UltraUART] ThreadUltraUART unavailable; ultrasonic safety disabled.", file=sys.stderr)
        return

    if ultra_thread is None or not ultra_thread.is_alive():
        ultra_thread = ThreadUltraUART(
            port=ULTRA_UART_PORT,
            baud=ULTRA_UART_BAUD,
            stale_s=ULTRA_STALE_SECONDS,
        )
        ultra_thread.start()
        print(f"[UltraUART] Start requested on {ULTRA_UART_PORT} @ {ULTRA_UART_BAUD}")


def stop_ultra_thread():
    """Stop ultrasonic UART reader thread."""
    global ultra_thread

    if ultra_thread is not None:
        try:
            ultra_thread.stop()
            ultra_thread.join(timeout=1.0)
        except Exception as e:
            print(f"[UltraUART] Stop error: {e}", file=sys.stderr)
        ultra_thread = None


def ultra_safety_worker():
    """Safety supervisor: obstacle => speed 0, clear => restore previous speed."""
    global ultra_safety_running, command_sender, sign_detection_controller, ultra_last_ui_speed, ultra_last_status

    ultra_blocking = False
    saved_speed_ui = None
    clear_since = None
    last_cmd_time = 0.0

    while ultra_safety_running:
        time.sleep(ULTRA_CHECK_INTERVAL)

        if not ultra_safety_enabled:
            continue

        if ultra_thread is None or not ultra_thread.is_alive():
            continue

        state = ultra_thread.get_state_copy()
        stale = ultra_thread.is_stale()
        now = time.time()
        age_ms = int((now - state.t) * 1000) if state.t > 0 else None

        with ultra_status_lock:
            ultra_last_status = {
                "ultra_blocking": ultra_blocking,
                "saved_speed_ui": saved_speed_ui,
                "stale": stale,
                "valid": bool(state.valid),
                "present": bool(state.present),
                "dist_cm": state.dist_cm,
                "last_update_age_ms": age_ms,
            }

        if not state.valid or stale:
            clear_since = None
            continue

        # Obstacle present: enter/maintain blocking
        if state.present:
            clear_since = None

            if not ultra_blocking:
                current_speed_ui = ultra_last_ui_speed
                if sign_detection_controller is not None:
                    try:
                        with sign_detection_controller.lock:
                            current_speed_ui = int(sign_detection_controller.current_speed)
                    except Exception:
                        pass

                # Save only non-zero speed, and only on block transition.
                if current_speed_ui > 0 and saved_speed_ui is None:
                    saved_speed_ui = current_speed_ui

                if now - last_cmd_time >= ULTRA_MIN_STOP_INTERVAL and command_sender is not None:
                    sent = command_sender.send_speed_command(0)
                    if sent:
                        ultra_blocking = True
                        last_cmd_time = now
                        print(
                            f"[UltraSafety] Blocking - obstacle at {state.dist_cm} cm, "
                            f"saved_speed={saved_speed_ui}"
                        )
            else:
                # Maintain stop with rate limit to avoid command spam.
                if now - last_cmd_time >= ULTRA_MIN_STOP_INTERVAL and command_sender is not None:
                    sent = command_sender.send_speed_command(0)
                    if sent:
                        last_cmd_time = now
            continue

        # No obstacle present, while currently blocking -> restore after stable clear delay
        if ultra_blocking:
            if clear_since is None:
                clear_since = now

            if (now - clear_since) < ULTRA_CLEAR_DELAY:
                continue

            restored_speed = saved_speed_ui
            if (
                restored_speed is not None
                and restored_speed > 0
                and command_sender is not None
                and (now - last_cmd_time) >= ULTRA_MIN_STOP_INTERVAL
            ):
                sent = command_sender.send_speed_command(int(restored_speed))
                if sent:
                    last_cmd_time = now
                    print(f"[UltraSafety] Restored speed to {restored_speed}")
                else:
                    print("[UltraSafety] Failed to restore speed command", file=sys.stderr)
            else:
                print(f"[UltraSafety] Clear path - no speed restoration needed (saved_speed={restored_speed})")

            ultra_blocking = False
            saved_speed_ui = None
            clear_since = None


def start_ultra_safety():
    """Start UltraSafety supervisor thread."""
    global ultra_safety_thread, ultra_safety_running

    if ultra_safety_thread is None or not ultra_safety_thread.is_alive():
        ultra_safety_running = True
        ultra_safety_thread = threading.Thread(target=ultra_safety_worker, daemon=True)
        ultra_safety_thread.start()
        print("[UltraSafety] Supervisor started")


def stop_ultra_safety():
    """Stop UltraSafety supervisor thread."""
    global ultra_safety_running, ultra_safety_thread

    ultra_safety_running = False
    if ultra_safety_thread:
        ultra_safety_thread.join(timeout=1.0)
        ultra_safety_thread = None
        print("[UltraSafety] Supervisor stopped")


def autopilot_status_broadcast_worker():
    """Background thread that broadcasts autopilot status to SSE clients."""
    global autopilot_status_broadcast_running, autopilot_controller, event_clients
    
    while autopilot_status_broadcast_running:
        try:
            status_data = None
            if autopilot_controller is not None:
                status = autopilot_controller.get_status()
                pid_params = autopilot_controller.get_pid_parameters()
                status_data = {
                    'type': 'autopilot_status',
                    'status': status,
                    'pid_parameters': pid_params,
                    'timestamp': time.time()
                }
            else:
                # Controller not initialized
                status_data = {
                    'type': 'autopilot_status',
                    'error': 'Auto-pilot controller not initialized',
                    'timestamp': time.time()
                }
            
            # Broadcast to unified event clients
            with event_clients_lock:
                for client_queue in event_clients[:]:
                    try:
                        client_queue.put(status_data, block=False)
                    except queue.Full:
                        # Client queue is full, remove it (client might be disconnected)
                        try:
                            event_clients.remove(client_queue)
                        except ValueError:
                            pass  # Already removed
            
            # Update every 0.5 seconds
            time.sleep(0.5)
        except Exception as e:
            if autopilot_status_broadcast_running:
                print(f"[AutopilotStatusBroadcast] Error: {e}", file=sys.stderr)
                time.sleep(0.5)


def start_autopilot_status_broadcast():
    """Start the autopilot status broadcast thread."""
    global autopilot_status_broadcast_thread, autopilot_status_broadcast_running
    
    if autopilot_status_broadcast_thread is None or not autopilot_status_broadcast_thread.is_alive():
        autopilot_status_broadcast_running = True
        autopilot_status_broadcast_thread = threading.Thread(
            target=autopilot_status_broadcast_worker, daemon=True)
        autopilot_status_broadcast_thread.start()


def stop_autopilot_status_broadcast():
    """Stop the autopilot status broadcast thread."""
    global autopilot_status_broadcast_running, autopilot_status_broadcast_thread
    
    autopilot_status_broadcast_running = False
    if autopilot_status_broadcast_thread:
        autopilot_status_broadcast_thread.join(timeout=1.0)
        autopilot_status_broadcast_thread = None


def sign_detection_status_broadcast_worker():
    """Background thread that broadcasts sign detection status to SSE clients."""
    global sign_detection_status_broadcast_running, sign_detection_controller, event_clients, sign_detector
    
    while sign_detection_status_broadcast_running:
        try:
            status_data = None
            if sign_detection_controller is not None:
                status = sign_detection_controller.get_status()
                
                # Get extra status from DETECTOR (frames, device, etc.)
                detector_status = {}
                detections = []
                if sign_detector:
                    detector_status = sign_detector.get_status()
                    detections = sign_detector.get_detections()
                
                # Merge detector status into main status object so frontend can see frame_count, device, etc.
                status.update(detector_status)
                
                status_data = {
                    'type': 'sign_detection_status',
                    'status': status,
                    'detections': detections,
                    'timestamp': time.time()
                }
            else:
                # Controller not initialized
                status_data = {
                    'type': 'sign_detection_status',
                    'error': 'Sign detection controller not initialized',
                    'timestamp': time.time()
                }
            
            # Broadcast to unified event clients
            with event_clients_lock:
                for client_queue in event_clients[:]:
                    try:
                        client_queue.put(status_data, block=False)
                    except queue.Full:
                        # Client queue is full, remove it (client might be disconnected)
                        try:
                            event_clients.remove(client_queue)
                        except ValueError:
                            pass  # Already removed
            
            # Update every 0.5 seconds
            time.sleep(0.5)
        except Exception as e:
            if sign_detection_status_broadcast_running:
                print(f"[SignDetectionStatusBroadcast] Error: {e}", file=sys.stderr)
                time.sleep(0.5)


def start_sign_detection_status_broadcast():
    """Start the sign detection status broadcast thread."""
    global sign_detection_status_broadcast_thread, sign_detection_status_broadcast_running
    
    if sign_detection_status_broadcast_thread is None or not sign_detection_status_broadcast_thread.is_alive():
        sign_detection_status_broadcast_running = True
        sign_detection_status_broadcast_thread = threading.Thread(
            target=sign_detection_status_broadcast_worker, daemon=True)
        sign_detection_status_broadcast_thread.start()


def stop_sign_detection_status_broadcast():
    """Stop the sign detection status broadcast thread."""
    global sign_detection_status_broadcast_running, sign_detection_status_broadcast_thread
    
    sign_detection_status_broadcast_running = False
    if sign_detection_status_broadcast_thread:
        sign_detection_status_broadcast_thread.join(timeout=1.0)
        sign_detection_status_broadcast_thread = None


@app.route('/uart/ports', methods=['GET'])
def list_ports():
    """List available serial ports."""
    if not SERIAL_AVAILABLE:
        return jsonify({'error': 'pyserial not installed', 'ports': []}), 500

    try:
        ports_list = list(serial_list_ports.comports())
        ports = []
        for p in ports_list:
            ports.append({
                'device': p.device,
                'description': p.description or 'No description',
                'hwid': getattr(p, 'hwid', 'Unknown')
            })
        # The WCH vendor CH341 driver registers /dev/ttyCH341USB* outside the
        # usb-serial bus, so pyserial never lists it. Add those nodes by name.
        import glob
        known = {p['device'] for p in ports}
        for dev in sorted(glob.glob('/dev/ttyCH341USB*')):
            if dev not in known:
                ports.append({
                    'device': dev,
                    'description': 'CH340/CH341 (WCH vendor driver) - ESP32',
                    'hwid': 'USB VID:PID=1A86:7523'
                })
        return jsonify({'ports': ports, 'count': len(ports)}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'ports': []}), 500


def _watchdog_safe_init(is_initialized):
    """Run a blocking controller initialization without tripping the ESP32 watchdog.

    Initializing the RealSense camera (autopilot) or loading the TensorRT model (sign
    detection) freezes Python for several hundred ms, so the heartbeat thread cannot
    send and the ESP32 watchdog (only active in AUTO) goes to FAULT. If the ESP32 is in
    AUTO: stop the car, switch to MANUAL while initializing, and restore AUTO after.
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            from command_sender import build_mode, build_speed
            if is_initialized() or system_state.get('mode') != 'AUTO' \
                    or not (serial_conn and serial_conn.is_open):
                return fn(*args, **kwargs)
            print(f"[Dashboard] {fn.__name__}: AUTO -> MANUAL during init (avoid watchdog FAULT)")
            write_uart_command(build_speed(0))
            write_uart_command(build_mode(False))
            try:
                return fn(*args, **kwargs)
            finally:
                write_uart_command(build_mode(True))
                print(f"[Dashboard] {fn.__name__}: init done, back to AUTO")
        return wrapper
    return decorator


@_watchdog_safe_init(lambda: autopilot_controller is not None)
def initialize_autopilot_if_needed():
    """Initialize autopilot controller if conditions are met."""
    global autopilot_controller, command_sender, video_streamer, serial_conn
    
    # Only initialize if not already initialized
    if autopilot_controller is not None:
        return True
    
    # Check if we have all required components
    if CommandSender is None or AutoPilotController is None or MarcosLaneDetector_Advanced is None:
        print("Autopilot modules not available", file=sys.stderr)
        return False
    
    if not serial_conn or not serial_conn.is_open:
        print("Serial port not connected", file=sys.stderr)
        return False
    
    # Initialize command sender
    if command_sender is None:
        command_sender = CommandSender(write_uart_command)
    
    # Initialize video streamer if not already done
    if video_streamer is None:
        # Try RealSense first if available
        if RealSenseStreamer is not None:
            print(f"[Dashboard] Attempting to initialize RealSense camera using: {RealSenseStreamer}")
            rs_streamer = RealSenseStreamer()
            if rs_streamer.initialize():
                video_streamer = rs_streamer
                print("[Dashboard] RealSense camera initialized successfully.")
            else:
                print("[Dashboard] RealSense initialization failed, falling back to standard VideoStreamer.")
        else:
            print("[Dashboard] RealSenseStreamer class not available (import failed).")
        
        # Fallback to standard VideoStreamer if RealSense failed or not available
        if video_streamer is None and VideoStreamer is not None:
            video_streamer = VideoStreamer()
            if not video_streamer.initialize():
                return False
    
    # Initialize autopilot controller
    if video_streamer is not None:
        if MarcosLaneDetector_Advanced is None:
            return False
        
        # Instantiate lane detector strategy
        lane_detector = MarcosLaneDetector_Advanced(threshold=180)
        
        # Initialize autopilot controller with default parameters
        autopilot_controller = AutoPilotController(
            video_streamer=video_streamer,
            command_sender=command_sender,
            lane_detector=lane_detector,
            pid_kp=0.8380,
            pid_ki=0.0100,
            pid_kd=0.4300,
            max_angle=30.0,
            deadband=6.0
        )
        
        # If sign controller exists, update its reference to autopilot
        global sign_detection_controller
        if sign_detection_controller is not None:
             sign_detection_controller.autopilot_controller = autopilot_controller
             print("[Dashboard] Linked AutopilotController to SignController")
             
        return True
    else:
        return False

def on_sign_controller_event(event_type, data):
    """Callback to handle events from SignController."""
    global event_clients, event_clients_lock
    
    if event_type == "sign_detected":
        message = f"[SignController] {data.get('message', 'Sign detected')}"
        
        # Create log message for dashboard (use specialized log type)
        log_data = {
            'type': 'server_log',
            'message': message,
            'timestamp': time.time()
        }
        
        # Broadcast
        with event_clients_lock:
            for client_queue in event_clients[:]:
                try:
                    client_queue.put(log_data, block=False)
                except:
                    pass


@_watchdog_safe_init(lambda: sign_detection_controller is not None)
def initialize_sign_detection_if_needed():
    """Initialize sign detection controller if conditions are met."""
    global sign_detection_controller, sign_detector, video_streamer, serial_conn, command_sender, autopilot_controller
    
    # Only initialize if not already initialized
    if sign_detection_controller is not None:
        return True
    
    # Check if we have all required components
    if SignDetector is None or SignController is None:
        print("Sign detection modules not available", file=sys.stderr)
        return False
    
    if not serial_conn or not serial_conn.is_open:
        print("Serial port not connected", file=sys.stderr)
        return False
        
    # Initialize command sender if needed
    if command_sender is None:
        command_sender = CommandSender(write_uart_command)
    
    # Initialize video streamer if not already done
    if video_streamer is None:
        # Try RealSense first if available
        if RealSenseStreamer is not None:
            print(f"[Dashboard] Attempting to initialize RealSense camera using: {RealSenseStreamer}")
            rs_streamer = RealSenseStreamer()
            if rs_streamer.initialize():
                video_streamer = rs_streamer
                print("[Dashboard] RealSense camera initialized successfully.")
            else:
                print("[Dashboard] RealSense initialization failed, falling back to standard VideoStreamer.")
        else:
            print("[Dashboard] RealSenseStreamer class not available (import failed).")
        
        # Fallback to standard VideoStreamer if RealSense failed or not available
        if video_streamer is None and VideoStreamer is not None:
            video_streamer = VideoStreamer()
            if not video_streamer.initialize():
                video_streamer = None
                return False
    
    # Create sign detection controller if video_streamer is available
    if video_streamer is not None:
        # Initialize the detector (sensor)
        if sign_detector is None:
            sign_detector = SignDetector(
                video_streamer=video_streamer,
                confidence_threshold=0.6
            )
            if not sign_detector.initialize():
                sign_detector = None
                return False
                
        # Initialize the controller (decision maker)
        sign_detection_controller = SignController(
            sign_detector=sign_detector,
            command_sender=command_sender,
            event_callback=on_sign_controller_event,
            autopilot_controller=autopilot_controller
        )
        
        return True
    else:
        return False

@app.route('/uart/connect', methods=['POST'])
def uart_connect():
    """Connect to Nucleo (STM32) via UART."""
    global serial_conn, serial_read_buffer, command_sender

    if not SERIAL_AVAILABLE:
        return jsonify({'error': 'pyserial not installed'}), 500

    data = request.get_json() or {}
    port = data.get('port')

    if not port:
        return jsonify({'error': 'Port not specified'}), 400

    # Already connected to this port: do nothing. Reopening the port resets the
    # ESP32 (the WCH CH340 driver toggles DTR/RTS anyway), which takes ~10 s and
    # drops it back to MANUAL/ARMED, losing AUTO mode and the commands sent meanwhile.
    if serial_conn and serial_conn.is_open and serial_conn.port == port:
        return jsonify({
            'status': 'ok',
            'message': f'Already connected to {port}',
            'port': port,
            'baudrate': UART_BAUD_RATE,
        })

    try:
        if serial_conn and serial_conn.is_open:
            stop_serial_reader()
            serial_conn.close()
            serial_read_buffer = ""  # Clear buffer on reconnect

        serial_conn = open_serial(port, UART_BAUD_RATE)
        serial_read_buffer = ""  # Clear buffer on new connection
        
        # Initialize command sender immediately after connection
        if command_sender is None:
            command_sender = CommandSender(write_uart_command)
            
        time.sleep(0.1)
        start_serial_reader()
        print("[Dashboard] Started serial reader")
        start_heartbeat_sender()
        print("[Dashboard] Started heartbeat sender")
        
        response = {
            'status': 'ok',
            'message': f'Connected to {port}',
            'port': port,
            'baudrate': UART_BAUD_RATE,
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/uart/disconnect', methods=['POST'])
def uart_disconnect():
    """Disconnect from Nucleo (UART)."""
    global serial_conn, serial_read_buffer, autopilot_controller, sign_detection_controller, sign_detector

    stop_serial_reader()
    stop_heartbeat_sender()
    
    # Stop autopilot if running
    if autopilot_controller:
        autopilot_controller.stop()
    
    # Stop sign detection if running
    if sign_detection_controller:
        sign_detection_controller.stop()
        # Also stop the detector (the thread)
        if sign_detector:
            sign_detector.stop()

    if serial_conn and serial_conn.is_open:
        serial_conn.close()
        serial_conn = None
    
    serial_read_buffer = ""  # Clear buffer on disconnect

    return jsonify({'status': 'ok', 'message': 'Disconnected'})


@app.route('/events')
def events_stream():
    """Unified Server-Sent Events stream for all system events."""
    # Start all broadcast threads if not already running
    start_serial_reader()
    start_autopilot_status_broadcast()
    start_sign_detection_status_broadcast()
    start_ultra_thread()
    start_ultra_safety()
    
    def generate():
        # Create a dedicated queue for this client
        client_queue = queue.Queue(maxsize=100)

        # Register this client
        with event_clients_lock:
            event_clients.append(client_queue)

        try:
            while True:
                try:
                    # Get message from this client's queue
                    msg = client_queue.get(timeout=1.0)
                    
                    # If message is a dict (status update), serialize it
                    if isinstance(msg, dict):
                        if msg.get('type') == 'serial':
                            # Serial messages are raw strings in 'message' field
                            message = msg.get('message', '')
                            if message and len(message.strip()) > 0:
                                # Compatibility: send raw serial line as 'data' for backward compatibility with serial listener
                                # BUT we should wrap it in JSON ideally. 
                                # For now, let's check what the client expects.
                                # The new client will parse JSON.
                                json_data = json.dumps(msg)
                                yield f"data: {json_data}\n\n"
                        else:
                            # Status updates are dicts
                            json_data = json.dumps(msg)
                            yield f"data: {json_data}\n\n"
                    else:
                         # Fallback for string messages
                         yield f"data: {msg}\n\n"

                except queue.Empty:
                    yield ": heartbeat\n\n"
        except GeneratorExit:
            pass  # Client disconnected
        except Exception as e:
            print(f"[EventsSSE] Error: {e}", file=sys.stderr)
        finally:
            # Unregister this client
            with event_clients_lock:
                try:
                    event_clients.remove(client_queue)
                except ValueError:
                    pass  # Already removed

    response = Response(stream_with_context(generate()),
                        mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'  # Disable buffering for nginx
    return response


# Deprecated streams - kept for compatibility if user hasn't refreshed page
# We redirect them to the unified logic or just keep them working but with reduced functionality

@app.route('/uart/stream')
def uart_stream_deprecated():
    """Deprecated: Use /events instead."""
    return events_stream()

@app.route('/autopilot/status/stream')
def autopilot_status_stream_deprecated():
    """Deprecated: Use /events instead."""
    return events_stream()

@app.route('/sign_detection/status/stream')
def sign_detection_status_stream_deprecated():
    """Deprecated: Use /events instead."""
    return events_stream()


@app.route('/status')
def get_status():
    """Get current system status (mode and state)."""
    global system_state
    with system_state_lock:
        return jsonify({
            'mode': system_state['mode'],
            'state': system_state['state']
        })


def generate_debug_mjpeg(image_key=None, mode='autopilot', 
                         waiting_text='Waiting...', not_running_text='Not running'):
    """
    Generate MJPEG stream from debug images.
    
    Args:
        image_key: Key for autopilot debug images (e.g., 'bird_view_lines', 'sliding_windows', 'final_result')
        mode: 'autopilot' or 'sign_detector'
        waiting_text: Text to show when waiting
        not_running_text: Text to show when controller is not running
    """
    import cv2
    import time
    import numpy as np
    
    # Declare globals explicitly
    global autopilot_controller, sign_detector
    
    # Create a placeholder black image
    placeholder_base = np.zeros((480, 640, 3), dtype=np.uint8)
    
    while True:
        debug_image = None
        is_running = False
        
        try:
            if mode == 'autopilot':
                if autopilot_controller is not None:
                    # Check status
                    try:
                        status = autopilot_controller.get_status()
                        is_running = status.get('is_running', False)
                    except:
                        is_running = False
                        
                    # Get image only if running
                    if is_running:
                        debug_image = autopilot_controller.get_debug_image(image_key)
            
            elif mode == 'sign_detector':
                if sign_detector is not None:
                    # Check status
                    try:
                        status = sign_detector.get_status()
                        is_running = status.get('is_running', False)
                    except:
                        is_running = False
                        
                    # Get image only if running
                    if is_running:
                        debug_image = sign_detector.get_detection_image()
                    
        except Exception as e:
            # print(f"[Debug Stream] Error fetching data: {e}")
            debug_image = None
            
        if debug_image is not None:
            # Make sure image is valid (not empty)
            try:
                if debug_image.size > 0 and len(debug_image.shape) >= 2:
                    # Convert grayscale ROI to BGR if needed for display
                    if len(debug_image.shape) == 2:  # Grayscale image (like 'roi' or 'binary')
                        debug_image = cv2.cvtColor(debug_image, cv2.COLOR_GRAY2BGR)
                    
                    ret, buffer = cv2.imencode('.jpg', debug_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    else:
                        # Encoding failed
                        debug_image = None
                else:
                    # Invalid image
                    debug_image = None
            except Exception as e:
                # print(f"[Debug Stream] Error processing image: {e}")
                debug_image = None
        
        # If we have no image (either fetch failed, or processing failed, or just not available)
        if debug_image is None:
            # Create appropriate placeholder text
            placeholder_text = not_running_text if not is_running else waiting_text
            
            placeholder = placeholder_base.copy()
            cv2.putText(placeholder, placeholder_text, (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS


DEBUG_SNAPSHOT_SOURCES = {
    'bird_view_lines': dict(mode='autopilot', image_key='bird_view_lines', waiting_text='Waiting for lane detection...', not_running_text='Lane detection not running'),
    'sliding_windows': dict(mode='autopilot', image_key='sliding_windows', waiting_text='Waiting for lane detection...', not_running_text='Lane detection not running'),
    'final_result': dict(mode='autopilot', image_key='final_result', waiting_text='Waiting for lane detection...', not_running_text='Lane detection not running'),
    'sign_detections': dict(mode='sign_detector', waiting_text='Waiting for sign detection...', not_running_text='Sign detection not running'),
}


@app.route('/debug/snapshot/<name>')
def debug_snapshot(name):
    """Single JPEG frame of a debug view.

    The dashboard polls these instead of keeping 4 MJPEG streams open: browsers allow
    only 6 HTTP/1.1 connections per host, and 4 streams + /events + /uart/stream used
    them all, so START/STOP requests stayed queued in the browser forever.
    """
    source = DEBUG_SNAPSHOT_SOURCES.get(name)
    if source is None:
        return jsonify({'error': f'Unknown debug view: {name}'}), 404
    gen = generate_debug_mjpeg(**source)
    try:
        chunk = next(gen)
    finally:
        gen.close()
    jpeg = chunk.split(b'\r\n\r\n', 1)[1].rstrip(b'\r\n')
    resp = Response(jpeg, mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.route('/debug/bird_view_lines')
def debug_bird_view_lines():
    """MJPEG stream for bird view with lines."""
    return Response(
        stream_with_context(generate_debug_mjpeg(mode='autopilot', image_key='bird_view_lines', waiting_text='Waiting for lane detection...', not_running_text='Lane detection not running')),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/debug/sliding_windows')
def debug_sliding_windows():
    """MJPEG stream for sliding windows."""
    return Response(
        stream_with_context(generate_debug_mjpeg(mode='autopilot', image_key='sliding_windows', waiting_text='Waiting for lane detection...', not_running_text='Lane detection not running')),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/debug/final_result')
def debug_final_result():
    """MJPEG stream for final result."""
    return Response(
        stream_with_context(generate_debug_mjpeg(mode='autopilot', image_key='final_result', waiting_text='Waiting for lane detection...', not_running_text='Lane detection not running')),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/autopilot/start', methods=['POST'])
def autopilot_start():
    """Start the auto-pilot controller."""
    global autopilot_controller
    
    # Try to initialize if not already initialized
    if autopilot_controller is None:
        # Quick checks before potentially blocking initialization
        if CommandSender is None or AutoPilotController is None or MarcosLaneDetector_Advanced is None:
            response = {
                'error': 'Auto-pilot controller not initialized',
                'details': {
                    'modules_available': {
                        'CommandSender': CommandSender is not None,
                        'AutoPilotController': AutoPilotController is not None,
                        'VideoStreamer': VideoStreamer is not None,
                        'MarcosLaneDetector_Advanced': MarcosLaneDetector_Advanced is not None
                    }
                },
                'suggestions': ['Autopilot modules not found - check that you are in brain/dashboard directory']
            }
            # Add import errors if available
            if import_errors:
                response['details']['import_errors'] = import_errors
            return jsonify(response), 503
        
        if not serial_conn or not serial_conn.is_open:
            return jsonify({
                'error': 'Auto-pilot controller not initialized',
                'details': {'serial_connected': False},
                'suggestions': ['Connect UART port first']
            }), 503
        
        # Initialize (may block briefly if camera is slow, but optimized to be fast)
        initialized = initialize_autopilot_if_needed()
        if not initialized:
            # Get detailed status for error message
            status = {
                'error': 'Auto-pilot controller not initialized',
                'details': {
                    'modules_available': {
                        'CommandSender': CommandSender is not None,
                        'AutoPilotController': AutoPilotController is not None,
                        'VideoStreamer': VideoStreamer is not None,
                        'MarcosLaneDetector_Advanced': MarcosLaneDetector_Advanced is not None
                    },
                    'serial_connected': serial_conn is not None and serial_conn.is_open if serial_conn else False,
                    'video_streamer_initialized': video_streamer is not None
                },
                'suggestions': []
            }
            
            # Add import errors if available
            if import_errors:
                status['details']['import_errors'] = import_errors
            
            if not status['details']['serial_connected']:
                status['suggestions'].append('Connect UART port first')
            if not status['details']['video_streamer_initialized']:
                if not status['details']['modules_available']['VideoStreamer']:
                    status['suggestions'].append('Install autopilot modules: pip install opencv-python numpy')
                else:
                    status['suggestions'].append('Connect camera USB device')
            if not status['details']['modules_available']['AutoPilotController'] or not status['details']['modules_available']['MarcosLaneDetector_Advanced']:
                status['suggestions'].append('Autopilot modules not found - check that you are in brain/dashboard directory')
            
            return jsonify(status), 503
    
    success = autopilot_controller.start()
    if success:
        # Restart sign detector if running to prevent stream freezing
        # This handles the case where starting Autopilot might interfere with an existing SignDetector loop
        if sign_detector is not None:
            try:
                status = sign_detector.get_status()
                if status.get('is_running', False):
                    print("Restarting SignDetector to prevent freeze...", file=sys.stderr)
                    sign_detector.stop()
                    time.sleep(0.1) # Brief pause to allow thread cleanup
                    sign_detector.start()
            except Exception as e:
                print(f"Error restarting SignDetector: {e}", file=sys.stderr)

        return jsonify({'status': 'ok', 'message': 'Auto-pilot started'})
    else:
        return jsonify({'error': 'Auto-pilot already running'}), 400

@app.route('/autopilot/stop', methods=['POST'])
def autopilot_stop():
    """Stop the auto-pilot controller."""
    global autopilot_controller, video_streamer, sign_detection_controller
    if autopilot_controller is None:
        return jsonify({'error': 'Auto-pilot controller not initialized'}), 503
    
    success = autopilot_controller.stop()
    if success:
        # Check if video_streamer should be stopped (only if sign_detection is also not running)
        if video_streamer is not None:
            sign_detection_running = False
            if sign_detection_controller is not None:
                sign_status = sign_detection_controller.get_status()
                sign_detection_running = sign_status.get('is_running', False)
            
            # If neither autopilot nor sign_detection is running, stop video_streamer
            if not sign_detection_running:
                print("Stopping video streamer (no controllers running)", file=sys.stderr)
                video_streamer.stop()
                video_streamer = None
                # Also set autopilot_controller to None to force re-initialization
                # This ensures video_streamer will be recreated when starting again
                autopilot_controller = None
                # IMPORTANT: Also clear sign detectors because they hold a reference to the now-stopped video_streamer
                sign_detection_controller = None
                sign_detector = None
        
        return jsonify({'status': 'ok', 'message': 'Auto-pilot stopped'})
    else:
        return jsonify({'error': 'Auto-pilot not running'}), 400

@app.route('/autopilot/pid', methods=['POST'])
def autopilot_update_pid():
    """Update PID parameters."""
    global autopilot_controller
    if autopilot_controller is None:
        return jsonify({'error': 'Auto-pilot controller not initialized'}), 503
    
    data = request.get_json() or {}
    kp = data.get('kp')
    ki = data.get('ki')
    kd = data.get('kd')
    max_angle = data.get('max_angle')
    # Handle both 'deadband' and 'tolerance' (dashboard sends 'tolerance')
    deadband = data.get('deadband')
    if deadband is None:
        deadband = data.get('tolerance')
    
    # Validate parameters
    if kp is not None and (not isinstance(kp, (int, float)) or kp < 0):
        return jsonify({'error': 'Invalid kp value'}), 400
    if ki is not None and (not isinstance(ki, (int, float)) or ki < 0):
        return jsonify({'error': 'Invalid ki value'}), 400
    if kd is not None and (not isinstance(kd, (int, float)) or kd < 0):
        return jsonify({'error': 'Invalid kd value'}), 400
    if max_angle is not None and (not isinstance(max_angle, (int, float)) or max_angle < 0):
        return jsonify({'error': 'Invalid max_angle value'}), 400
    if deadband is not None and (not isinstance(deadband, (int, float)) or deadband < 0):
        return jsonify({'error': 'Invalid deadband value'}), 400
    
    # Update parameters
    autopilot_controller.update_pid_parameters(kp=kp, ki=ki, kd=kd, max_angle=max_angle, deadband=deadband)
    
    # Return updated parameters
    updated_params = autopilot_controller.get_pid_parameters()
    return jsonify({
        'status': 'ok',
        'message': 'PID parameters updated',
        'pid_parameters': updated_params
    })

@app.route('/autopilot/pid', methods=['GET'])
def autopilot_get_pid():
    """Get current PID parameters."""
    global autopilot_controller
    if autopilot_controller is None:
        return jsonify({'error': 'Auto-pilot controller not initialized'}), 503
    
    pid_params = autopilot_controller.get_pid_parameters()
    return jsonify(pid_params)


@app.route('/autopilot/memory_mode', methods=['GET'])
def autopilot_get_memory_mode():
    """Get lane detector memory mode state."""
    global autopilot_controller
    if autopilot_controller is None:
        return jsonify({'error': 'Auto-pilot controller not initialized'}), 503

    lane_detector = getattr(autopilot_controller, 'lane_detector', None)
    if lane_detector is None or not hasattr(lane_detector, 'ENABLE_MEMORY_MODE'):
        return jsonify({'error': 'Lane detector memory mode not available'}), 400

    return jsonify({'enabled': bool(lane_detector.ENABLE_MEMORY_MODE)})


@app.route('/autopilot/memory_mode', methods=['POST'])
def autopilot_set_memory_mode():
    """Enable/disable lane detector memory mode."""
    global autopilot_controller
    if autopilot_controller is None:
        return jsonify({'error': 'Auto-pilot controller not initialized'}), 503

    data = request.get_json() or {}
    enabled = data.get('enabled')
    if not isinstance(enabled, bool):
        return jsonify({'error': 'Invalid enabled value, expected boolean'}), 400

    lane_detector = getattr(autopilot_controller, 'lane_detector', None)
    if lane_detector is None or not hasattr(lane_detector, 'ENABLE_MEMORY_MODE'):
        return jsonify({'error': 'Lane detector memory mode not available'}), 400

    lane_detector.ENABLE_MEMORY_MODE = enabled
    return jsonify({'status': 'ok', 'enabled': bool(lane_detector.ENABLE_MEMORY_MODE)})


@app.route('/sign_detection/start', methods=['POST'])
def sign_detection_start():
    """Start the sign detection controller."""
    global sign_detection_controller, sign_detector
    
    # Try to initialize if not already initialized
    if sign_detection_controller is None:
        initialized = initialize_sign_detection_if_needed()
        if not initialized:
            # Get detailed status for error message
            status = {
                'error': 'Sign detection controller not initialized',
                'details': {
                    'modules_available': {
                        'SignDetector': SignDetector is not None,
                        'SignController': SignController is not None,
                        'VideoStreamer': VideoStreamer is not None
                    },
                    'serial_connected': serial_conn is not None and serial_conn.is_open if serial_conn else False,
                    'video_streamer_initialized': video_streamer is not None
                },
                'suggestions': []
            }
            
            # Add import errors if available
            if import_errors:
                status['details']['import_errors'] = import_errors
            
            if not status['details']['serial_connected']:
                status['suggestions'].append('Connect UART port first')
            if not status['details']['video_streamer_initialized']:
                if not status['details']['modules_available']['VideoStreamer']:
                    status['suggestions'].append('Install autopilot modules: pip install opencv-python numpy')
                else:
                    status['suggestions'].append('Connect camera USB device')
            if not status['details']['modules_available']['SignDetector']:
                status['suggestions'].append('Sign detection modules not found - check that sign_vision module is available')
            
            return jsonify(status), 503
    
    # Start both the detector (eye) and controller (brain)
    detector_success = sign_detector.start()
    controller_success = sign_detection_controller.start()
    
    if detector_success or controller_success:
        return jsonify({'status': 'ok', 'message': 'Sign detection started'})
    else:
        return jsonify({'error': 'Sign detection already running'}), 400

@app.route('/sign_detection/stop', methods=['POST'])
def sign_detection_stop():
    """Stop the sign detection controller."""
    global sign_detection_controller, video_streamer, autopilot_controller, sign_detector
    if sign_detection_controller is None:
        return jsonify({'error': 'Sign detection controller not initialized'}), 503
    
    # Stop both
    controller_success = sign_detection_controller.stop()
    detector_success = sign_detector.stop()
    
    if controller_success or detector_success:
        # Check if video_streamer should be stopped (only if autopilot is also not running)
        if video_streamer is not None:
            autopilot_running = False
            if autopilot_controller is not None:
                autopilot_status = autopilot_controller.get_status()
                autopilot_running = autopilot_status.get('is_running', False)
            
            # If neither autopilot nor sign_detection is running, stop video_streamer
            if not autopilot_running:
                print("Stopping video streamer (no controllers running)", file=sys.stderr)
                video_streamer.stop()
                video_streamer = None
                # Also set global variables to None to force re-initialization
                # This ensures video_streamer will be recreated when starting again
                sign_detection_controller = None
                sign_detector = None
                # IMPORTANT: Also clear autopilot_controller because it holds a reference to the now-stopped video_streamer
                autopilot_controller = None
        
        return jsonify({'status': 'ok', 'message': 'Sign detection stopped'})
    else:
        return jsonify({'error': 'Sign detection not running'}), 400

@app.route('/sign_detection/confidence', methods=['POST'])
def sign_detection_update_confidence():
    """Update confidence threshold."""
    global sign_detection_controller, sign_detector
    if sign_detection_controller is None:
        return jsonify({'error': 'Sign detection controller not initialized'}), 503
    
    data = request.get_json() or {}
    threshold = data.get('threshold')
    
    if threshold is None or not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
        return jsonify({'error': 'Invalid threshold value (must be between 0.0 and 1.0)'}), 400
    
    # Update the DETECTOR directly
    if sign_detector:
        sign_detector.update_confidence_threshold(float(threshold))
        status = sign_detector.get_status()
        return jsonify({
            'status': 'ok',
            'message': 'Confidence threshold updated',
            'confidence_threshold': status['confidence_threshold']
        })
    else:
        return jsonify({'error': 'Sign detector not initialized'}), 500


@app.route('/debug/sign_detections')
def debug_sign_detections():
    """MJPEG stream for sign detections."""
    return Response(
        stream_with_context(generate_debug_mjpeg(
            mode='sign_detector',
            waiting_text='Waiting for sign detection...',
            not_running_text='Sign detection not running'
        )),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/health')
def health():
    """Health check endpoint."""
    global serial_conn, serial_reader_running, video_streamer, autopilot_controller, sign_detection_controller, ultra_last_status
    port_name = None
    if serial_conn and serial_conn.is_open:
        port_name = serial_conn.port
    
    autopilot_status_info = None
    if autopilot_controller is not None:
        autopilot_status_info = autopilot_controller.get_status()
    
    sign_detection_status_info = None
    if sign_detection_controller is not None:
        sign_detection_status_info = sign_detection_controller.get_status()

    with ultra_status_lock:
        ultra_status_info = dict(ultra_last_status)
    
    return jsonify({
        'status': 'ok',
        'message': 'Dashboard server is running',
        'uart_connected': serial_conn is not None and serial_conn.is_open if serial_conn else False,
        'uart_port': port_name,
        'serial_available': SERIAL_AVAILABLE,
        'serial_reader_running': serial_reader_running,
        'video_streamer_initialized': video_streamer is not None,
        'autopilot_status': autopilot_status_info,
        'sign_detection_status': sign_detection_status_info,
        'ultra_safety_enabled': ultra_safety_enabled,
        'ultra_uart_alive': ultra_thread is not None and ultra_thread.is_alive(),
        'ultra_safety_alive': ultra_safety_thread is not None and ultra_safety_thread.is_alive(),
        'ultra_status': ultra_status_info
    })


def get_local_ip():
    """Return the primary local IP address for display (e.g. in dashboard URL)."""
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(('10.255.255.255', 1))
            return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'


def select_port_interactive() -> str:
    """Interactive port selection from available serial ports."""
    if not SERIAL_AVAILABLE:
        print("Warning: pyserial not installed. Cannot select port.")
        return None

    ports = list(serial_list_ports.comports())
    if not ports:
        # The WCH vendor CH341 driver registers /dev/ttyCH341USB* outside the
        # usb-serial bus, so pyserial does not list it. Look for it by name.
        import glob
        extra = sorted(glob.glob('/dev/ttyCH341USB*'))
        if extra:
            print(f"pyserial found no ports; using CH341 device: {extra[0]}")
            return extra[0]
        print("No serial ports found. Connect the ESP32 / Nucleo and try again.")
        return None
    if len(ports) == 1:
        selected = ports[0].device
        print(f"Only one port found, auto-selecting: {selected}")
        return selected
    print("\nAvailable serial ports:")
    for idx, p in enumerate(ports):
        print(f"  [{idx}] {p.device} - {p.description}")
    while True:
        sel = input("\nSelect port index (or press Enter to skip): ")
        try:
            if sel.strip() == "":
                return None  # Skip selection

            index = int(sel)
            if 0 <= index < len(ports):
                return ports[index].device
            else:
                print("Invalid index, out of range. Please select a valid port index.")
        except ValueError:
            print("Invalid input, please enter a number or press Enter to skip.")


def _main():
    """Entry point for CLI: connect UART if requested, then run Flask."""
    global serial_conn
    import argparse

    parser = argparse.ArgumentParser(description='Flask dashboard server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--port-name', dest='port_name', default=None, help='Serial port to connect to (e.g. /dev/ttyACM0)')
    parser.add_argument('--auto-connect', dest='auto_connect', action='store_true', help='Interactively select serial port on startup')
    parser.add_argument('--lane-detection', dest='lane_detection', action='store_true', help='Start lane detection (autopilot) when UART connected')
    parser.add_argument('--sign-detection', dest='sign_detection', action='store_true', help='Start sign detection when UART connected')
    parser.add_argument('--protocol', choices=SUPPORTED_PROTOCOLS, default=get_uart_protocol(),
                        help='Low-level controller UART protocol (default: %(default)s, or env AURA_UART_PROTOCOL)')

    args = parser.parse_args()
    set_uart_protocol(args.protocol)
    local_ip = get_local_ip()

    print("=" * 70)
    print(" " * 15 + f"Car Control Dashboard (UART protocol: {get_uart_protocol()})")
    print("=" * 70)
    print()

    uart_connected = False
    print("📡 Serial Port Connection")
    print("-" * 70)
    if args.port_name:
        if SERIAL_AVAILABLE:
            try:
                print(f"   Connecting to {args.port_name}...", end=" ")
                serial_conn = open_serial(args.port_name, UART_BAUD_RATE)
                time.sleep(0.1)
                start_serial_reader()
                start_heartbeat_sender()
                print(f"✓ Connected at {UART_BAUD_RATE} baud")
                uart_connected = True
            except Exception as e:
                print(f"✗ Failed: {e}")
                print("   You can connect manually from the dashboard.")
        else:
            print("   ✗ pyserial not installed. Cannot connect to serial port.")
    elif args.auto_connect:
        if SERIAL_AVAILABLE:
            selected_port = select_port_interactive()
            if selected_port:
                try:
                    print(f"   Connecting to {selected_port}...", end=" ")
                    serial_conn = open_serial(selected_port, UART_BAUD_RATE)
                    time.sleep(0.1)
                    start_serial_reader()
                    start_heartbeat_sender()
                    print(f"✓ Connected at {UART_BAUD_RATE} baud")
                    uart_connected = True
                except Exception as e:
                    print(f"✗ Failed: {e}")
                    print("   You can connect manually from the dashboard.")
            else:
                print("   ⚠ No port selected. You can connect manually from the dashboard.")
        else:
            print("   ✗ pyserial not installed. Cannot connect to serial port.")
    else:
        print("   ⚠ Not connected. Use the dashboard to connect manually.")
        print("   Tip: Use --auto-connect to select a port on startup, or --port-name to specify one.")
    print()

    print("🛰️  UltraSafety Initialization")
    print("-" * 70)
    start_ultra_thread()
    start_ultra_safety()
    print("   ℹ Ultra UART reader and safety supervisor requested.")
    print()

    if uart_connected:
        print("🚗 Controller Initialization")
        print("-" * 70)
        if args.lane_detection:
            print("   [1/2] Lane Detection (Autopilot)")
            initialized = initialize_autopilot_if_needed()
            if initialized:
                success = autopilot_controller.start()
                if success:
                    print("      ✓ Autopilot controller initialized and started")
                else:
                    print("      ✗ Failed to start autopilot controller")
            else:
                print("      ✗ Autopilot initialization failed - check camera and modules")
        if args.sign_detection:
            print("   [2/2] Sign Detection")
            initialized = initialize_sign_detection_if_needed()
            if initialized:
                detector_success = sign_detector.start()
                controller_success = sign_detection_controller.start()
                if detector_success and controller_success:
                    print("      ✓ Sign detection controller initialized and started")
                else:
                    print("      ✗ Failed to start sign detection (detector: {}, controller: {})".format(
                        "✓" if detector_success else "✗",
                        "✓" if controller_success else "✗"
                    ))
            else:
                print("      ✗ Sign detection initialization failed - check camera and modules")
        if args.lane_detection or args.sign_detection:
            print()
    else:
        print("🚗 Controller Initialization")
        print("-" * 70)
        if VideoStreamer is None:
            print("   ⚠ Video streamer not available - autopilot modules not found")
        else:
            print("   ℹ Controllers will be initialized when started from the dashboard.")
        print()

    print("=" * 70)
    print("🌐 Dashboard Server")
    print("-" * 70)
    print(f"   Local:   http://127.0.0.1:{args.port}")
    print(f"   Network: http://{local_ip}:{args.port}")
    print()
    print("   Press Ctrl+C to stop the server")
    print("=" * 70)
    print()
    print(f"Starting Dashboard Server on {args.host}:{args.port}")
    print(f"Access the dashboard at: http://localhost:{args.port}")
    print("Press Ctrl+C to stop the server")

    try:
        app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    except KeyboardInterrupt:
        print("\nServer stopped by user")


if __name__ == '__main__':
    _main()
