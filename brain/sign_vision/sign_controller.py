"""
Sign Controller Module - Orchestrates sign detection and command sending.

Following SRP: This module orchestrates the interaction between the sign detector and the command sender.
"""

import threading
import time
import sys
from pathlib import Path

# Add brain directory to path to import shared modules
brain_dir = Path(__file__).parent.parent
if str(brain_dir) not in sys.path:
    sys.path.insert(0, str(brain_dir))

from command_sender import CommandSender
from sign_vision.sign_detector import SignDetector
from .strategies import (
    DefaultStopStrategy,
    EnterIntersectionStrategy,
    IncreaseSpeedAndLaneWidthStrategy,
    CrosswalkStrategy,
    ParkingStrategy,
    ChangeSpeedStrategy,
)


class SignController:
    """Orchestrates sign detection and command sending."""
    DEFAULT_RESUME_SPEED = 120

    def __init__(self, sign_detector: SignDetector, command_sender: CommandSender, event_callback=None, autopilot_controller=None):
        """
        Initialize the sign controller.
        
        Args:
            sign_detector: SignDetector instance for detecting signs
            command_sender: CommandSender instance for sending commands
            event_callback: Optional callback function(event_type, data) for reporting events
            autopilot_controller: Optional AutoPilotController instance to pause/resume lane following
        """
        self.sign_detector = sign_detector
        self.command_sender = command_sender
        self.event_callback = event_callback
        self.autopilot_controller = autopilot_controller
        
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # State tracking
        self.last_command = None
        self.command_count = 0
        self.error_count = 0
        self.current_speed = 0          # Velocidad actual del carro. Se actualiza desde el dashboard y desde las estrategias.
        self.last_speed_before_stop = 0 # Última velocidad no-cero antes de un stop/slowdown.
        self.pending_resume_at = None
        self.pending_resume_speed = None
        # Bumped by cancel_pending_motion(). Delayed actions (strategy threads)
        # take a copy before sleeping and do nothing if it changed meanwhile.
        self.motion_epoch = 0
        
        # Cooldowns (to prevent spamming commands)
        self.last_stop_time = 0
        self.stop_cooldown = 5.0  # Seconds before stopping again
        
        # Initialize strategies (keys must match detector class names)
        self.strategies = {
            'stop': DefaultStopStrategy(self, self.lock),
            'end-autobahn': ChangeSpeedStrategy(self, self.lock, target_speed=50),
            'start-autobahn': ChangeSpeedStrategy(self, self.lock, target_speed=200),
            'priority-road': EnterIntersectionStrategy(self, self.lock),
            'one-way': IncreaseSpeedAndLaneWidthStrategy(self, self.lock),
            'pedestrian': CrosswalkStrategy(self, self.lock),
            'park': ParkingStrategy(self, self.lock),
            'roundabout-mini': EnterIntersectionStrategy(self, self.lock),
        }
        
    def start(self):
        """Start the sign controller."""
        with self.lock:
            if self.is_running:
                return False
            
            self.is_running = True
            self.pending_resume_at = None
            self.pending_resume_speed = None
            self.thread = threading.Thread(target=self._control_loop, daemon=True)
            self.thread.start()
            print("[SignController] Started")
            return True
    
    def stop(self):
        """Stop the sign controller."""
        with self.lock:
            if not self.is_running:
                return False
            
            self.is_running = False
            self.pending_resume_at = None
            self.pending_resume_speed = None
            print("[SignController] Stopping...")
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            print("[SignController] Stopped")
        
        return True

    def cancel_pending_motion(self, reason: str) -> None:
        """
        Drop every speed the strategies have scheduled and abort running maneuvers.

        Call when the car must stay stopped: brake, stop, disarm, FAULT, reboot.
        The ESP32 brake is one-shot (no lock-out), so a resume scheduled before
        the brake would otherwise move the car again a few seconds later.
        """
        with self.lock:
            had_resume = self.pending_resume_at is not None
            self.pending_resume_at = None
            self.pending_resume_speed = None
            self.motion_epoch += 1
        aborted = [name for name, s in self.strategies.items()
                   if hasattr(s, "abort") and s.abort(reason)]
        if had_resume or aborted:
            print(f"[SignController] Cancelled pending motion ({reason}): "
                  f"resume={had_resume} aborted={sorted(set(aborted))}")

    def on_obstacle_detected(self) -> None:
        """The ESP32 blocked forward motion. Let maneuvers that need it give up."""
        for name, s in self.strategies.items():
            if hasattr(s, "on_obstacle_detected"):
                s.on_obstacle_detected()

    def schedule_speed_resume(self, delay_seconds: float, override_speed: int = None) -> int:
        """
        Schedule a speed restore after a stop or slowdown.

        Args:
            delay_seconds: Seconds to wait before restoring speed.
            override_speed: If provided, resume at this exact speed instead of last_speed_before_stop.
        """
        delay_seconds = max(0.0, float(delay_seconds))
        with self.lock:
            if override_speed is not None:
                resume_speed = int(max(0, min(255, override_speed)))
            elif self.last_speed_before_stop > 0:
                resume_speed = self.last_speed_before_stop
            else:
                resume_speed = self.DEFAULT_RESUME_SPEED
            self.pending_resume_speed = resume_speed
            self.pending_resume_at = time.time() + delay_seconds
            return resume_speed
    
    def _control_loop(self):
        """Main control loop running in background thread."""
        while True:
            resume_speed_to_send = None
            with self.lock:
                if not self.is_running:
                    break
                if (
                    self.pending_resume_at is not None
                    and self.pending_resume_speed is not None
                    and time.time() >= self.pending_resume_at
                ):
                    resume_speed_to_send = self.pending_resume_speed
                    self.pending_resume_at = None
                    self.pending_resume_speed = None
                    self.last_command = f"speed:{resume_speed_to_send} (resume)"

            if resume_speed_to_send is not None:
                sent = self.command_sender.send_speed_command(resume_speed_to_send)
                if sent:
                    with self.lock:
                        self.current_speed = resume_speed_to_send
                if self.event_callback:
                    self.event_callback("speed_resumed", {
                        "speed": resume_speed_to_send,
                        "sent": bool(sent)
                    })
            
            try:
                # 1. Get latest detections
                detections = self.sign_detector.get_detections()
                
                # 2. Process detections and decide action
                action_taken = False
                
                # Sort by confidence (highest first)
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                
                for detection in detections:
                    label = detection['class'].lower()
                    confidence = detection['confidence']
                    
                    # Use dynamic confidence threshold from detector
                    threshold = self.sign_detector.confidence_threshold
                    
                    # --- LOGIC BASED ON PROVIDED CLASSES ---
                    
                    # Check if we have a strategy for this label
                    if label in self.strategies and confidence >= threshold:
                        strategy = self.strategies[label]
                        if strategy.execute(detection):
                            action_taken = True
                            break # Priority action taken
                        
                if action_taken:
                    with self.lock:
                        self.command_count += 1
                
                # Control loop rate (10Hz is usually enough for signs)
                time.sleep(0.1)
                
            except Exception as e:
                print(f"[SignController] Error in control loop: {e}")
                with self.lock:
                    self.error_count += 1
                time.sleep(1.0)

    def get_status(self) -> dict:
        """Get current status of the sign controller."""
        with self.lock:
            resume_in_seconds = None
            if self.pending_resume_at is not None:
                resume_in_seconds = max(0.0, self.pending_resume_at - time.time())
            return {
                'is_running': self.is_running,
                'command_count': self.command_count,
                'error_count': self.error_count,
                'last_command': self.last_command,
                'current_speed': self.current_speed,
                'last_speed_before_stop': self.last_speed_before_stop,
                'resume_pending': self.pending_resume_at is not None,
                'resume_speed': self.pending_resume_speed,
                'resume_in_seconds': resume_in_seconds
            }

    def update_current_speed(self, speed: int):
        """Update current speed. Always reflects the real speed of the car.
        Saves the last positive speed into last_speed_before_stop before setting to 0."""
        with self.lock:
            if self.current_speed > 0 and speed == 0:
                self.last_speed_before_stop = self.current_speed
            self.current_speed = speed
