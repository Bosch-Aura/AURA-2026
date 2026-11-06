#!/usr/bin/env python3
"""
Flask server for the car control dashboard (ESP32 over UART).
Serves the dashboard and proxies commands to the ESP32 via serial.
Protocol: Bosch-Aura/embedded docs/BRAIN_TEAM_PROTOCOL.md
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

# UART Configuration (ESP32 USB link; the board shows up as /dev/ttyCH341USB0 on the Jetson)
UART_BAUD_RATE = 115200
# Opening the port reboots the ESP32: wait this long for EVENT:SYSTEM_READY before arming anyway
ESP32_BOOT_TIMEOUT = 3.0

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
# The ESP32 boots DISARMED and has no driving modes: 'mode' is tracked only on the Jetson side
system_state = {'mode': 'MANUAL', 'state': 'DISARMED'}
system_state_lock = threading.Lock()
esp32_ready_event = threading.Event()  # Set on EVENT:SYSTEM_READY

# Auto-pilot components
video_streamer = None
autopilot_controller = None
command_sender = None

# Sign vision components
sign_detector = None
sign_detection_controller = None
sign_detection_initializing = False  # True while loading YOLO model in background

# UltraSafety: the ESP32 stops by itself on an obstacle (STOPPED) and reports it with
# EVENT:OBSTACLE_DETECTED / EVENT:OBSTACLE_CLEARED. On clear it does NOT resume the
# latched speed, so the Jetson re-sends the last requested speed.
ultra_safety_enabled = True
ultra_last_status = {
    "ultra_blocking": False,
    "dist_cm": None,
    "last_event_age_ms": None,
}
ultra_last_event_time = None
ultra_status_lock = threading.Lock()

app = Flask(__name__, static_folder=SCRIPT_DIR)
CORS(app)  # Enable CORS for all routes


def write_uart_command(command: str):
    """Write a UART command line (CHANNEL:COMMAND:VALUE\\n) to the ESP32."""
    global serial_conn
    if not serial_conn or not serial_conn.is_open:
        return False, "Serial port not connected"

    try:
        with serial_lock:
            if not command.endswith("\n"):
                command = command + "\n"
            line = command.encode("utf-8")
            serial_conn.write(line)
            serial_conn.flush()
            print(f"[UART TX] {line.decode('utf-8', errors='replace').strip()!r}")
            return True, "OK"
    except Exception as e:
        return False, str(e)


def translate_http_to_uart(endpoint: str, args: dict):
    """
    Translate a dashboard HTTP endpoint to an ESP32 action.

    Returns (success, message, action). action is a function that takes the
    CommandSender and returns True if sent, or None when nothing must be sent.
    Speed/steer go through CommandSender so direction tracking stays consistent.
    """
    SERVO_CENTER = 105
    SERVO_RIGHT = 50   # Right turn (lower value)
    SERVO_LEFT = 160   # Left turn (higher value)
    SPEED_UI_MAX = 255
    FORWARD_SPEED = 220

    if endpoint == 'arm':
        return True, "ARM", lambda cs: cs.send_arm()
    elif endpoint == 'disarm':
        return True, "DISARM", lambda cs: cs.send_disarm()
    elif endpoint == 'mode':
        # The ESP32 has no driving modes: AUTO/MANUAL only lives on the Jetson
        value = args.get('value', '').upper()
        if value not in ('AUTO', 'MANUAL'):
            return False, "Invalid mode value", None
        with system_state_lock:
            system_state['mode'] = value
        return True, f"MODE: {value}", None
    elif endpoint == 'brake':
        return True, "BRAKE", lambda cs: cs.send_brake()
    elif endpoint == 'forward':
        return True, "FORWARD", lambda cs: cs.send_speed_command(FORWARD_SPEED, "forward")
    elif endpoint == 'back':
        return True, "BACKWARD", lambda cs: cs.send_speed_command(SPEED_UI_MAX, "backward")
    elif endpoint == 'driveStop':
        return True, "STOP", lambda cs: cs.send_speed_command(0)
    elif endpoint == 'changeSpeed':
        speed_str = args.get('speed', '0')
        direction_str = args.get('direction', 'forward')
        try:
            speed = int(speed_str)
            if speed < 0 or speed > SPEED_UI_MAX:
                return False, f"Speed must be 0-{SPEED_UI_MAX}", None
            if speed == 0:
                return True, "STOP", lambda cs: cs.send_speed_command(0)
            return (
                True,
                f"SPEED: {speed} ({direction_str})",
                lambda cs: cs.send_speed_command(speed, direction_str),
            )
        except ValueError:
            return False, "Invalid speed value", None
    elif endpoint == 'steer' or endpoint == 'changeSteer':
        angle_str = args.get('angle', '105')
        try:
            angle = int(angle_str)
            if angle < SERVO_RIGHT or angle > SERVO_LEFT:
                return False, f"Angle must be {SERVO_RIGHT}-{SERVO_LEFT}", None
            return True, f"STEER: {angle}°", lambda cs: cs.send_steering_command(angle)
        except ValueError:
            return False, "Invalid angle value", None
    elif endpoint == 'left':
        return True, "LEFT", lambda cs: cs.send_steering_command(SERVO_LEFT)
    elif endpoint == 'right':
        return True, "RIGHT", lambda cs: cs.send_steering_command(SERVO_RIGHT)
    elif endpoint == 'steerStop':
        return True, "CENTER", lambda cs: cs.send_steering_command(SERVO_CENTER)
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
    """Handle all dashboard commands and translate to ESP32 commands."""
    args = request.args.to_dict()

    success, message, action = translate_http_to_uart(endpoint, args)

    if not success:
        return jsonify({'error': message}), 400

    if action is None:
        return jsonify({'status': 'ok', 'message': message}), 200

    if command_sender is None or not serial_conn or not serial_conn.is_open:
        return jsonify({'error': 'Failed to send UART command: Serial port not connected'}), 500

    if action(command_sender):
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

        return jsonify({'status': 'ok', 'message': message}), 200
    else:
        return jsonify({'error': 'Failed to send UART command'}), 500


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

    ser = serial.Serial(
        port=port,
        baudrate=baud,
        timeout=0.1,  # Shorter timeout for more responsive reading
        write_timeout=1.0,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        xonxoff=False,
        rtscts=False,
        dsrdtr=False
    )
    ser.reset_input_buffer()
    ser.reset_output_buffer()
    return ser


def arm_when_ready():
    """Fallback arm: if opening the port did not reboot the ESP32, no SYSTEM_READY arrives."""
    if esp32_ready_event.wait(ESP32_BOOT_TIMEOUT):
        return  # parse_system_events already armed on EVENT:SYSTEM_READY
    print(f"[Dashboard] No EVENT:SYSTEM_READY after {ESP32_BOOT_TIMEOUT}s, arming anyway")
    if command_sender is not None:
        command_sender.send_arm()


def connect_serial(port: str):
    """Open the ESP32 port, start reader + heartbeat and arm once the board has booted."""
    global serial_conn, serial_read_buffer, command_sender

    if serial_conn and serial_conn.is_open:
        stop_serial_reader()
        serial_conn.close()

    esp32_ready_event.clear()
    serial_conn = open_serial(port, UART_BAUD_RATE)
    serial_read_buffer = ""  # Clear buffer on new connection

    # Initialize command sender immediately after connection
    if command_sender is None:
        command_sender = CommandSender(write_uart_command)
    command_sender.reset_link_state()

    time.sleep(0.1)
    start_serial_reader()
    print("[Dashboard] Started serial reader")
    start_heartbeat_sender()
    print("[Dashboard] Started heartbeat sender")
    threading.Thread(target=arm_when_ready, daemon=True).start()


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
    """Parse ESP32 protocol events (EVENT:...) for state tracking. Debug lines are ignored."""
    global system_state, ultra_last_event_time

    if line.startswith('EVENT:STATE_CHANGED:'):
        # Extract only the first word to avoid trailing garbage/extra messages
        raw_state = line.replace('EVENT:STATE_CHANGED:', '').strip()
        state = raw_state.split()[0] if raw_state else raw_state
        
        with system_state_lock:
            previous = system_state['state']
            system_state['state'] = state
            print(f"[SystemState] State changed to: {state} (raw: {raw_state})")

        # DISARMED and FAULT center the servo on the ESP32, but the autopilot
        # only sends C:SET_STEER when its angle changes, so after re-arming it
        # would never re-send it. Forget the last angle when the car becomes
        # drivable again: the next autopilot cycle re-sends it.
        if state == 'RUNNING' and previous not in ('RUNNING', 'STOPPED'):
            if autopilot_controller is not None:
                autopilot_controller.invalidate_steering()
    elif line.startswith('EVENT:SYSTEM_READY'):
        # The board (re)booted: nothing is latched and it is DISARMED. Arm it again,
        # like the ROS2 bridge does; nothing moves until a new C:SET_SPEED arrives.
        print("[SystemState] ESP32 ready")
        esp32_ready_event.set()
        if command_sender is not None:
            command_sender.reset_link_state()
            command_sender.send_arm()
    elif line.startswith('EVENT:WATCHDOG_TIMEOUT') or line.startswith('EVENT:ESTOP_TRIGGERED'):
        # FAULT is never cleared automatically: re-arm from the dashboard
        print(f"[SystemState] ESP32 fault: {line}", file=sys.stderr)
    elif line.startswith('EVENT:OBSTACLE_DETECTED'):
        raw_dist = line.replace('EVENT:OBSTACLE_DETECTED', '').lstrip(':').strip()
        with ultra_status_lock:
            ultra_last_event_time = time.time()
            ultra_last_status['ultra_blocking'] = True
            ultra_last_status['dist_cm'] = int(raw_dist) if raw_dist.isdigit() else None
        print(f"[UltraSafety] ESP32 stopped - obstacle at {raw_dist} cm")
    elif line.startswith('EVENT:OBSTACLE_CLEARED'):
        with ultra_status_lock:
            ultra_last_event_time = time.time()
            ultra_last_status['ultra_blocking'] = False
            ultra_last_status['dist_cm'] = None
        # The ESP32 cleared the latched speed when it stopped: re-send the last one
        # the brain asked for (it may have changed while the car was held).
        if ultra_safety_enabled and command_sender is not None:
            if command_sender.resend_last_speed():
                print(f"[UltraSafety] Clear path - restored speed {command_sender.last_speed}")
            else:
                print("[UltraSafety] Clear path - no speed restoration needed")


def serial_reader_worker():
    """Background thread that reads from serial port."""
    global serial_conn, serial_reader_running, event_clients

    while serial_reader_running:
        if serial_conn and serial_conn.is_open:
            try:
                lines = read_available_lines(serial_conn)
                for line in lines:
                    if line:
                        # Parse system events for state tracking
                        parse_system_events(line)

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
    """Send M:PING:0 to the ESP32 so its link watchdog (300 ms) does not FAULT (every 50ms when UART connected)."""
    global serial_conn, heartbeat_sender_running, command_sender

    while heartbeat_sender_running:
        if serial_conn and serial_conn.is_open and command_sender:
            try:
                command_sender.send_heartbeat()
            except Exception as e:
                print(f"[Heartbeat] Error: {e}", file=sys.stderr)
        time.sleep(0.05)  # 50ms (20 Hz); ESP32 watchdog is 300 ms

    print("[Heartbeat] Worker thread stopped")


def start_heartbeat_sender():
    """Start the heartbeat sender thread."""
    global heartbeat_sender_thread, heartbeat_sender_running

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
        return jsonify({'ports': ports, 'count': len(ports)}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'ports': []}), 500


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
    """Connect to the ESP32 via UART."""
    if not SERIAL_AVAILABLE:
        return jsonify({'error': 'pyserial not installed'}), 500

    data = request.get_json() or {}
    port = data.get('port')

    if not port:
        return jsonify({'error': 'Port not specified'}), 400

    try:
        connect_serial(port)

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
    """Disconnect from the ESP32 (UART)."""
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
        # Leave the ESP32 safe instead of waiting for its watchdog to FAULT
        if command_sender is not None:
            command_sender.send_disarm()
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
    global serial_conn, serial_reader_running, video_streamer, autopilot_controller, sign_detection_controller
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
        if ultra_last_event_time is not None:
            ultra_status_info['last_event_age_ms'] = int((time.time() - ultra_last_event_time) * 1000)
    
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
        'esp32_state': system_state['state'],
        'ultra_safety_enabled': ultra_safety_enabled,
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
        print("No serial ports found. Connect the ESP32 (USB) and try again.")
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
    parser.add_argument('--port-name', dest='port_name', default=None, help='Serial port to connect to (e.g. /dev/ttyCH341USB0 on the Jetson)')
    parser.add_argument('--auto-connect', dest='auto_connect', action='store_true', help='Interactively select serial port on startup')
    parser.add_argument('--lane-detection', dest='lane_detection', action='store_true', help='Start lane detection (autopilot) when UART connected')
    parser.add_argument('--sign-detection', dest='sign_detection', action='store_true', help='Start sign detection when UART connected')

    args = parser.parse_args()
    local_ip = get_local_ip()

    print("=" * 70)
    print(" " * 15 + "Car Control Dashboard (ESP32)")
    print("=" * 70)
    print()

    uart_connected = False
    print("📡 Serial Port Connection")
    print("-" * 70)
    if args.port_name:
        if SERIAL_AVAILABLE:
            try:
                print(f"   Connecting to {args.port_name}...", end=" ")
                connect_serial(args.port_name)
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
                    connect_serial(selected_port)
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
