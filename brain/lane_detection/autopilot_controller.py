"""
AutoPilot Controller Module - Orchestrates lane detection, angle conversion, and command sending.

Following SRP: This module orchestrates the interaction between detector, converter, and sender.
"""

import threading
import time

# Import lane following modules (relative imports since we're in lane_detection)
from .lane_detector import LaneDetector
from .pid_controller import PIDController
from .filter_controller import FilterController
from .angle_converter import AngleConverter
from .intersection_event_controller import IntersectionEventController
from .intersection_strategies import FirstIntersectionLeftTurnStrategy
# Import shared resources (absolute imports - parent dir is in sys.path)
from command_sender import CommandSender
from camera.video_streamer import VideoStreamer


class AutoPilotController:
    """Orchestrates lane detection, angle conversion, and command sending."""
    
    def __init__(self, video_streamer: VideoStreamer, command_sender: CommandSender,
                 lane_detector: LaneDetector,
                 pid_kp: float = 0.06, pid_ki: float = 0.002, 
                 pid_kd: float = 0.02, max_angle: float = 30.0, deadband: float = 6.0,
                 max_change: float = 20.0, intersection_event_callback=None):
        """
        Initialize the auto-pilot controller.
        
        Args:
            video_streamer: VideoStreamer instance for getting frames
            command_sender: CommandSender instance for sending commands
            lane_detector: LaneDetector strategy instance for lane detection
            pid_kp: PID proportional gain
            pid_ki: PID integral gain
            pid_kd: PID derivative gain
            max_angle: Maximum steering angle in degrees (default: 30.0)
            deadband: Deadband angle in degrees (default: 6.0)
            max_change: Maximum allowed change in angle deviation between frames (default: 20.0)
            intersection_event_callback: Optional callback(event_type, data) for lane-based intersection events
        """
        self.video_streamer = video_streamer
        self.command_sender = command_sender
        self.lane_detector = lane_detector
        self.angle_converter = AngleConverter()
        self.filter_controller = FilterController(max_change=max_change)
        
        # Initialize PID controller
        self.pid_controller = PIDController(
            Kp=pid_kp,
            Ki=pid_ki,
            Kd=pid_kd,
            max_angle=max_angle,
            deadband=deadband
        )
        
        # Store PID parameters for later updates
        self.pid_kp = pid_kp
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        self.max_angle = max_angle
        self.deadband = deadband
        
        # Time tracking for PID dt calculation
        self.last_time = time.time()
        
        self.is_running = False
        self.is_paused = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Statistics
        self.last_steering_angle = None
        self.last_servo_angle = None
        self.command_count = 0
        self.error_count = 0
        
        # Debug images storage
        self.last_debug_images = None

        # Lane-based intersection event handling (separate from sign_controller)
        self.intersection_event_controller = IntersectionEventController(
            strategy=FirstIntersectionLeftTurnStrategy(
                autopilot_controller=self,
                event_callback=intersection_event_callback,
                duration_s=3.0,
                max_angle_ratio=0.75,
            ),
            consecutive_frames=3,
            cooldown_s=2.5,
        )

        # Timed steering override (used by lane-based intersection strategies)
        self._manual_override_until = 0.0
        self._manual_override_steering_angle = 0.0
        self._manual_override_label = None
    
    def start(self):
        """Start the auto-pilot controller."""
        with self.lock:
            if self.is_running:
                return False
            
            self.is_running = True
            self.last_debug_images = None  # Clear stale images
            self.thread = threading.Thread(target=self._control_loop, daemon=True)
            self.thread.start()
            print("[AutoPilotController] Started")
            return True
    
    def stop(self):
        """Stop the auto-pilot controller."""
        with self.lock:
            if not self.is_running:
                return False
            
            self.is_running = False
            print("[AutoPilotController] Stopping...")
        
        # Wait for the control loop thread to finish (with timeout)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                print("[AutoPilotController] Warning: Control loop thread did not stop within timeout")
            else:
                print("[AutoPilotController] Stopped")
        
        return True
    
    def pause(self):
        """Pause the auto-pilot controller (stop processing but keep thread alive)."""
        with self.lock:
            self.is_paused = True
            print("[AutoPilotController] Paused")

    def resume(self):
        """Resume the auto-pilot controller."""
        with self.lock:
            self.is_paused = False
            # Reset controllers to avoid jumps
            self.pid_controller.reset()
            self.filter_controller.reset()
            # Whoever paused us (a maneuver, the dashboard) may have moved the
            # servo: re-send the angle even if the PID gives the same value.
            self.last_servo_angle = None
            print("[AutoPilotController] Resumed")

    def _send_servo_if_changed(self, steering_angle: float, servo_angle: int) -> None:
        """Send C:SET_STEER only when the servo value changes (prevents UART spam)."""
        # Claim the value under the lock BEFORE sending: if invalidate_steering()
        # runs while the command is going out, its None is not overwritten.
        with self.lock:
            self.last_steering_angle = steering_angle  # always, for the dashboard
            if self.last_servo_angle == servo_angle:
                return
            self.last_servo_angle = servo_angle
        success = self.command_sender.send_steering_command(servo_angle)
        with self.lock:
            if success:
                self.command_count += 1
            else:
                self.error_count += 1

    def invalidate_steering(self) -> None:
        """
        Forget the last servo value sent, so the next cycle re-sends it.

        Call when the ESP32 re-arms (DISARMED/FAULT -> RUNNING): it centered the
        servo meanwhile, and the change-only filter above would otherwise never
        send the angle again while the PID output stays the same.
        """
        with self.lock:
            self.last_servo_angle = None

    def start_timed_steering_override(self, steering_angle_deg: float, duration_s: float, label: str = 'manual_override'):
        """Apply a temporary steering override without stopping the autopilot thread."""
        duration_s = max(0.0, float(duration_s))
        with self.lock:
            self._manual_override_steering_angle = float(steering_angle_deg)
            self._manual_override_until = time.time() + duration_s
            self._manual_override_label = str(label)

        # Reset controllers to avoid abrupt PID accumulation after override.
        self.pid_controller.reset()
        self.filter_controller.reset()

    def _control_loop(self):
        """Main control loop running in background thread."""
        while True:
            # Check if still running at the start of each iteration
            with self.lock:
                if not self.is_running:
                    break
            
            try:
                # Get frame from video streamer
                frame = self.video_streamer.get_frame()
                
                # Check again after getting frame (stop might have been called)
                with self.lock:
                    if not self.is_running:
                        break
                
                if frame is None:
                    time.sleep(0.033)  # Wait ~30ms if no frame
                    continue

                # Get lane metrics from detector strategy
                # We run this even if paused to keep debug images active
                angle_deviation_deg, debug_images = self.lane_detector.get_lane_metrics(frame)
                
                # Store debug images for streaming
                with self.lock:
                    if debug_images is not None:
                        self.last_debug_images = debug_images

                # Process lane-based intersection events via dedicated controller/strategy
                if debug_images is not None:
                    intersection_state = debug_images.get('intersection_state', {'detected': False})
                    self.intersection_event_controller.process(intersection_state)
                
                # Check if paused
                is_paused = False
                with self.lock:
                    is_paused = self.is_paused

                if is_paused:
                    time.sleep(0.033)
                    continue

                # Manual steering override (e.g., first intersection one-shot strategy)
                override_active = False
                override_angle = 0.0
                with self.lock:
                    now_ts = time.time()
                    if now_ts < self._manual_override_until:
                        override_active = True
                        override_angle = float(self._manual_override_steering_angle)
                    elif self._manual_override_until > 0:
                        self._manual_override_until = 0.0
                        self._manual_override_label = None

                if override_active:
                    steering_angle = override_angle
                    servo_angle = self.angle_converter.convert(steering_angle, inverted=True)
                    self._send_servo_if_changed(steering_angle, servo_angle)
                    time.sleep(0.033)
                    continue
                
                # Handle no lanes detected
                
                # Handle no lanes detected
                if angle_deviation_deg is None:
                    self.pid_controller.reset()
                    self.filter_controller.reset()
                    time.sleep(0.033)
                    continue
                
                # Filter outliers
                filtered_angle = self.filter_controller.filter(angle_deviation_deg)
                if filtered_angle is None:
                    print(f"[Filter] Rejected outlier: {angle_deviation_deg}")
                    time.sleep(0.033)
                    continue
                    
                angle_deviation_deg = filtered_angle
                
                # Calculate dt for PID
                current_time = time.time()
                dt = current_time - self.last_time
                if dt <= 0:
                    dt = 0.033  # Default to ~30 FPS if dt is invalid
                self.last_time = current_time
                
                # Check if still running before processing and sending commands
                with self.lock:
                    if not self.is_running:
                        break
                
                # Compute PID output (negate deviation for PID error)
                # Positive deviation (lane to right) → positive steering (turn right)
                pid_error = -angle_deviation_deg
                #print(f"Original angle deviation: {angle_deviation_deg}")
                steering_angle = self.pid_controller.compute(pid_error, dt)
                #print(f"PID Controller: Steering angle: {steering_angle}")
                
                # Check again before sending command (stop might have been called)
                with self.lock:
                    if not self.is_running:
                        break
                
                # Convert steering angle to servo angle
                servo_angle = self.angle_converter.convert(steering_angle, inverted=True)
                #print(f"Angle Converter: Servo angle: {servo_angle}")
                
                # Send command to ESP32 only if it changed (prevent UART spam)
                self._send_servo_if_changed(steering_angle, servo_angle)
                
                # Control loop rate (~30 FPS)
                time.sleep(0.033)
                
            except Exception as e:
                print(f"[AutoPilotController] Error in control loop: {e}")
                self.error_count += 1
                time.sleep(0.1)
    
    def get_status(self) -> dict:
        """Get current status of the auto-pilot controller."""
        with self.lock:
            return {
                'is_running': self.is_running,
                'last_steering_angle': self.last_steering_angle,
                'last_servo_angle': self.last_servo_angle,
                'command_count': self.command_count,
                'error_count': self.error_count,
                'intersection': self.intersection_event_controller.last_state,
                'manual_override': {
                    'active': time.time() < self._manual_override_until,
                    'label': self._manual_override_label,
                    'remaining_s': max(0.0, self._manual_override_until - time.time())
                }
            }
    
    def update_pid_parameters(self, kp: float = None, ki: float = None, kd: float = None, 
                             max_angle: float = None, deadband: float = None):
        """
        Update PID parameters dynamically.
        
        Args:
            kp: New proportional gain (None to keep current)
            ki: New integral gain (None to keep current)
            kd: New derivative gain (None to keep current)
            max_angle: New max angle (None to keep current)
            deadband: New deadband (None to keep current)
        """
        with self.lock:
            if kp is not None:
                self.pid_kp = kp
            if ki is not None:
                self.pid_ki = ki
            if kd is not None:
                self.pid_kd = kd
            if max_angle is not None:
                self.max_angle = max_angle
            if deadband is not None:
                self.deadband = deadband
            
            # Update PID controller parameters
            self.pid_controller.set_parameters(
                Kp=kp, Ki=ki, Kd=kd, max_angle=max_angle, deadband=deadband
            )
    
    def get_pid_parameters(self) -> dict:
        """Get current PID parameters."""
        with self.lock:
            pid_params = self.pid_controller.get_parameters()
            return {
                'kp': pid_params['Kp'],
                'ki': pid_params['Ki'],
                'kd': pid_params['Kd'],
                'max_angle': pid_params['max_angle'],
                'deadband': pid_params['deadband']
            }
    
    def get_debug_image(self, image_key: str):
        """
        Get a debug image by key.
        
        Args:
            image_key: Key of the debug image ('bird_view_lines', 'sliding_windows', 'final_result', etc.)
            
        Returns:
            Image (numpy array) or None if not available
        """
        with self.lock:
            if self.last_debug_images and image_key in self.last_debug_images:
                return self.last_debug_images[image_key].copy()
        return None

