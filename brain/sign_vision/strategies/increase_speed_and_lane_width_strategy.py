import time
import threading
from .base_strategy import SignStrategy

class IncreaseSpeedAndLaneWidthStrategy(SignStrategy):
    """Strategy for increasing speed to 230 and lane width to 550."""
    
    def __init__(self, controller, lock, cooldown=5.0, min_confidence=0.7, activation_distance=1.0):
        super().__init__(controller, lock, min_confidence, activation_distance)
        self.last_activation_time = 0
        self.cooldown = cooldown

    def _reset_speed_after_delay(self, delay=3.0):
        """Reset speed and lane parameters after a delay."""
        epoch = self.controller.motion_epoch
        print(f"[IncreaseSpeedAndLaneWidthStrategy] Waiting {delay}s before reset...")
        time.sleep(delay)
        if self.controller.motion_epoch != epoch:
            print("[IncreaseSpeedAndLaneWidthStrategy] Cancelled (brake/stop meanwhile): speed not reset")
            return
        
        print("[IncreaseSpeedAndLaneWidthStrategy] Resetting parameters...")
        try:
            # 1. Reset Speed to 210 (Normal Speed)
            RESET_SPEED = 230
            success = self.controller.command_sender.send_speed_command(RESET_SPEED)
            if success:
                # We need to lock when updating shared state from a thread
                with self.lock:
                    self.controller.current_speed = RESET_SPEED
            print(f"[IncreaseSpeedAndLaneWidthStrategy] Reset speed to {RESET_SPEED}")

            # 2. Reset Lane Width and Lookahead - DISABLED as per request
            # We only reset speed, keeping the wider lane parameters for now
            # if hasattr(self.controller, 'autopilot_controller') and self.controller.autopilot_controller:
            #     if hasattr(self.controller.autopilot_controller, 'lane_detector'):
            #         RESET_LANE_WIDTH = 500
            #         RESET_LOOKAHEAD = 250
            #         
            #         self.controller.autopilot_controller.lane_detector.LANE_WIDTH_PX = RESET_LANE_WIDTH
            #         self.controller.autopilot_controller.lane_detector.LOOKAHEAD_DISTANCE = RESET_LOOKAHEAD
            #         print(f"[IncreaseSpeedAndLaneWidthStrategy] Reset LANE_WIDTH_PX to {RESET_LANE_WIDTH}, LOOKAHEAD to {RESET_LOOKAHEAD}")
        except Exception as e:
            print(f"[IncreaseSpeedAndLaneWidthStrategy] Error during reset: {e}")

    def execute(self, detection: dict) -> bool:
        """
        Execute logic for going forward and widening lane expectation.
        Steps:
        1. Set speed to 230
        2. Set lane width expectation to 550px
        """
        # Check base conditions (confidence, distance)
        if not self.validate_detection(detection):
            return False
            
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_activation_time < self.cooldown:
            return False
            
        label = detection['class'].lower()
        confidence = detection['confidence']
        
        msg = f"{label.upper()} DETECTED! ({confidence:.2f}) - Increasing Speed & Lane Width"
        print(f"[IncreaseSpeedAndLaneWidthStrategy] {msg}")
        
        # Report event
        if self.controller.event_callback:
            self.controller.event_callback("sign_detected", {
                "label": label, 
                "confidence": float(confidence), 
                "message": msg
            })
            
        # Execute Action
        try:
            # 1. Increase Speed
            SPEED = 235
            print(f"[IncreaseSpeedAndLaneWidthStrategy] Setting speed to {SPEED}")
            
            success = self.controller.command_sender.send_speed_command(SPEED)
            if success:
                self.controller.current_speed = SPEED
            
            # 2. Increase Lane Width - not used for now
            # We need to access the autopilot controller -> lane detector
            if hasattr(self.controller, 'autopilot_controller') and self.controller.autopilot_controller:
                if hasattr(self.controller.autopilot_controller, 'lane_detector'):
                    # The lane detector stores LANE_WIDTH_PX. We assume it's public or accessible.
                    # Based on user context: self.LANE_WIDTH_PX = 500
                    TARGET_LANE_WIDTH = 500
                    self.controller.autopilot_controller.lane_detector.LANE_WIDTH_PX = TARGET_LANE_WIDTH
                    
                    # Also decrease lookahead distance for tighter response at speed
                    TARGET_LOOKAHEAD = 200
                    self.controller.autopilot_controller.lane_detector.LOOKAHEAD_DISTANCE = TARGET_LOOKAHEAD
                    print(f"[IncreaseSpeedAndLaneWidthStrategy] Set LANE_WIDTH_PX to {TARGET_LANE_WIDTH}, LOOKAHEAD to {TARGET_LOOKAHEAD}")
                else:
                    print("[IncreaseSpeedAndLaneWidthStrategy] Warning: Autopilot has no lane_detector")
            else:
                 print("[IncreaseSpeedAndLaneWidthStrategy] Warning: Autopilot controller not available, cannot update lane width")
                
        except Exception as e:
            print(f"[IncreaseSpeedAndLaneWidthStrategy] Error executing action: {e}")
        
        # Start background thread to reset parameters after 3 seconds
        reset_thread = threading.Thread(target=self._reset_speed_after_delay, args=(3.0,), daemon=True)
        reset_thread.start()

        self.last_activation_time = time.time()
        return True

