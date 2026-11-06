import threading
import time
import math

from .base_strategy import SignStrategy

# Servo angle constants (must match command_sender.py / angle_converter.py)
_SERVO_CENTER = 105
_SERVO_LEFT = 160
_SERVO_RIGHT = 50


class ParkingStrategy(SignStrategy):
    """
    Strategy for parallel parking to the right using a timed phase sequence.

    ## Parking space and vehicle dimensions
        Vehicle:        38 cm long × 21 cm wide
        Parking space:  80 cm long (Y) × 35 cm wide (X)

    ## Coordinate system (origin = rear-right corner of vehicle at maneuver start)
        +X  → into the parking space (lateral)
        +Y  → into the parking space (longitudinal / depth)

    ## Speed conversion
        All motion speeds are defined in cm/s and converted to UI units at
        construction time using:

            speed_ui = round(speed_cm_per_s / ui_speed_to_cm_per_s)

        The default conversion factor (ui_speed_to_cm_per_s = 0.3) was
        calibrated empirically: the car travelled ~450 cm in 80 s at
        speed_ui = 1, giving 450 / (80 × 1) = 5.625 cm/s per UI unit at
        low speed.  For maneuver speeds a factor of 0.3 is used.

    ## Phase geometry
        The classic three-arc parallel-parking maneuver is used:

        FORWARD_LEFT — advance with full-left steering for forward_left_distance_cm
            Angles the car toward the parking space entrance before reversing.

        BACK_RIGHT  — reverse with full-right steering for back_right_distance_cm
            Enters the space diagonally.

        BACK_LEFT   — reverse with full-left steering for back_left_distance_cm
            Mirrors BACK_RIGHT to straighten the car inside the space.

        After these three arcs the vehicle is parallel to the space walls and
        fully inside the 80 × 35 cm rectangle.

    ## Phase sequence
        WAIT          → Autopilot drives; duration = wait_target_distance_cm /
                        current_speed_cm_per_s so the car travels exactly
                        wait_target_distance_cm before the maneuver starts.
        BRAKE1        → Hold stop for brake_duration s.
        FORWARD_LEFT  → Advance full-left for forward_left_distance_cm to
                        angle the car toward the space entrance.
        BACK_RIGHT    → Reverse full-right for back_right_distance_cm.
        BACK_LEFT     → Reverse full-left  for back_left_distance_cm.
        STOP          → Hold stop for stop_hold_duration s, then finish.

    A cooldown prevents re-triggering while the maneuver is in progress.
    """

    def __init__(
        self,
        controller,
        lock,
        cooldown: float = 30.0,
        min_confidence: float = 0.6,
        activation_distance: float = 3.0,
        # --- speed parameters (all in cm/s, converted to UI internally) ---
        maneuver_speed_cm_per_s: float = 10.0,
        # --- speed conversion factor ---
        ui_speed_to_cm_per_s: float = 0.3,
        # --- wait phase ---
        wait_target_distance_cm: float = 220.0,
        default_wait_duration: float = 8.0,
        wait_crawl_speed_cm_per_s: float = 6.0,
        # --- maneuver arc distances (cm) ---
        forward_left_distance_cm: float = 20.0,
        back_right_distance_cm: float = 95.0,
        back_left_distance_cm: float = 80.0,
        # --- fixed durations (s) ---
        brake_duration: float = 0.5,
        stop_hold_duration: float = 1.0,
    ):
        """
        Args:
            controller: The SignController instance.
            lock: Threading lock shared with the controller.
            cooldown: Minimum seconds between activations.
            min_confidence: Minimum detection confidence to react.
            activation_distance: Maximum distance (m) at which the sign triggers.
            maneuver_speed_cm_per_s: Speed in cm/s used for all active motion
                phases. Converted to UI internally.
            ui_speed_to_cm_per_s: Calibration factor — how many cm/s correspond
                to 1 UI speed unit. Default 0.3 (empirically measured).
            wait_target_distance_cm: Distance in cm the car should travel during
                WAIT before the maneuver starts. Duration is computed as
                wait_target_distance_cm / current_speed_cm_per_s.
            default_wait_duration: Fallback WAIT duration (s) when the car is
                stopped at detection time.
            wait_crawl_speed_cm_per_s: Speed in cm/s sent to the car when it is
                stopped at detection time so it advances to the maneuver position.
            forward_left_distance_cm: Arc length (cm) for the FORWARD_LEFT phase.
                The car advances with full-left steering to angle itself toward
                the parking space before reversing.
            back_right_distance_cm: Arc length (cm) for the BACK_RIGHT phase.
                The car reverses with full-right steering to enter the space.
            back_left_distance_cm: Arc length (cm) for the BACK_LEFT phase.
                Mirrors BACK_RIGHT to straighten the car inside the space.
            brake_duration: Duration (s) of the BRAKE1 full-stop phase.
            stop_hold_duration: Duration (s) of the final STOP phase; ensures
                the ESP32 receives the stop command before the thread exits.
        """
        super().__init__(controller, lock, min_confidence, activation_distance)
        self.cooldown = float(cooldown)
        self.last_activation_time: float = 0.0

        # Speed conversion
        self.ui_speed_to_cm_per_s = float(ui_speed_to_cm_per_s)

        # Convert all speeds from cm/s to UI units
        self.maneuver_speed_ui   = max(1, round(maneuver_speed_cm_per_s / self.ui_speed_to_cm_per_s))
        self.wait_crawl_speed_ui = max(1, round(wait_crawl_speed_cm_per_s / self.ui_speed_to_cm_per_s))

        # Compute phase durations from arc distances and maneuver speed
        maneuver_speed_cm_per_s    = float(maneuver_speed_cm_per_s)
        self.forward_left_duration = float(forward_left_distance_cm) / maneuver_speed_cm_per_s
        self.back_right_duration   = float(back_right_distance_cm)   / maneuver_speed_cm_per_s
        self.back_left_duration    = float(back_left_distance_cm)    / maneuver_speed_cm_per_s

        # Fixed durations
        self.brake_duration     = float(brake_duration)
        self.stop_hold_duration = float(stop_hold_duration)

        # Wait phase parameters
        self.wait_target_distance_cm = float(wait_target_distance_cm)
        self.default_wait_duration   = float(default_wait_duration)

        # wait_duration is computed dynamically in execute()
        self.wait_duration: float = self.default_wait_duration

        # State machine (protected by self.lock)
        self.phase: str              = "IDLE"
        self.phase_start_time: float = 0.0
        self.is_running: bool        = False
        self.is_parked: bool         = False
        # Set by abort(): from then on the worker may only send speed 0.
        self._aborted: bool          = False

        self._worker_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public interface expected by SignController
    # ------------------------------------------------------------------

    def execute(self, detection: dict) -> bool:
        """
        Start the parking maneuver if conditions are met.

        Non-blocking: launches a background thread and returns immediately.

        Returns:
            True if the maneuver was started, False otherwise.
        """
        if not self.validate_detection(detection):
            return False

        now = time.time()

        if now - self.last_activation_time < self.cooldown:
            print("[ParkingStrategy] Cooldown active — ignoring parking sign.")
            return False

        with self.lock:
            if self.is_running:
                print("[ParkingStrategy] Maneuver already in progress — ignoring new trigger.")
                return False

            # Compute wait_duration so the car travels ~wait_target_distance_cm.
            # Formula: t = d / v  where v is current speed converted to cm/s.
            current_speed_ui = self.controller.current_speed
            current_speed_cm_per_s = current_speed_ui * self.ui_speed_to_cm_per_s

            if current_speed_cm_per_s > 0:
                self.wait_duration = self.wait_target_distance_cm / current_speed_cm_per_s
            else:
                # Car is stopped: send crawl speed and compute duration from it.
                self.controller.command_sender.send_speed_command(self.wait_crawl_speed_ui)
                crawl_speed_cm_per_s = self.wait_crawl_speed_ui * self.ui_speed_to_cm_per_s
                self.wait_duration = self.wait_target_distance_cm / crawl_speed_cm_per_s

            self.is_running = True
            self.is_parked = False
            self._aborted = False
            self.phase = "WAIT"
            self.phase_start_time = now

        # Keep autopilot active during WAIT so lane-following continues
        if hasattr(self.controller, "autopilot_controller") and self.controller.autopilot_controller:
            self.controller.autopilot_controller.resume()

        label = detection["class"].lower()
        confidence = detection["confidence"]
        msg = (
            f"{label.upper()} DETECTED! ({confidence:.2f}) "
            f"- Beginning parking countdown ({self.wait_duration:.1f}s)"
        )
        print(f"[ParkingStrategy] {msg}")

        if self.controller.event_callback:
            self.controller.event_callback(
                "sign_detected",
                {
                    "label": label,
                    "confidence": float(confidence),
                    "message": msg,
                },
            )

        self._worker_thread = threading.Thread(
            target=self._run_parking_sequence,
            name="ParkingStrategyWorker",
            daemon=True,
        )
        self._worker_thread.start()

        self.last_activation_time = now
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def abort(self, reason: str) -> bool:
        """
        Stop the maneuver now (brake, stop, disarm, FAULT, obstacle). Returns True
        if a maneuver was running. The worker exits on its next cycle; until then
        _set_motion() refuses anything but speed 0, so no command slips through
        after the brake. Lane following is resumed if parking had paused it.
        """
        with self.lock:
            if not self.is_running:
                return False
            paused_autopilot = self.phase != "WAIT"
            self.is_running = False
            self._aborted = True
            self.phase = "IDLE"
        print(f"[ParkingStrategy] Aborted ({reason})")
        if paused_autopilot and getattr(self.controller, "autopilot_controller", None):
            self.controller.autopilot_controller.resume()
        return True

    def on_obstacle_detected(self) -> None:
        """
        The ESP32 blocks forward motion while an obstacle is ahead, but the phase
        timers keep running: a blocked WAIT or FORWARD_LEFT would let the reverse
        arcs start from the wrong place. Give up instead of parking crooked.
        Reverse phases are not affected (backing away is allowed).
        """
        with self.lock:
            phase = self.phase if self.is_running else None
        if phase in ("WAIT", "FORWARD_LEFT"):
            self.abort(f"obstacle during {phase}")

    def _set_motion(self, speed_ui: int, direction: str, servo_angle: int) -> None:
        """
        Send a combined speed + steering command and update controller state.

        Args:
            speed_ui: Speed value in UI scale (0–255).
            direction: "forward" or "backward".
            servo_angle: Servo position (SERVO_RIGHT=50, CENTER=105, SERVO_LEFT=160).
        """
        speed_ui = max(0, min(255, int(speed_ui)))
        direction = "backward" if direction == "backward" else "forward"
        with self.lock:
            if self._aborted and speed_ui > 0:
                return

        ok_speed = self.controller.command_sender.send_speed_command(
            speed_ui, direction=direction
        )
        ok_steer = self.controller.command_sender.send_steering_command(servo_angle)

        with self.lock:
            if ok_speed:
                self.controller.current_speed = speed_ui
                self.controller.last_command = (
                    f"parking: speed={speed_ui} dir={direction} servo={servo_angle}"
                )
            if ok_speed and speed_ui > 0 and direction == "forward":
                self.controller.last_speed_before_stop = speed_ui

        if not ok_speed:
            print(f"[ParkingStrategy] Warning: failed to send speed command (speed={speed_ui}, dir={direction})")
        if not ok_steer:
            print(f"[ParkingStrategy] Warning: failed to send steering command (servo={servo_angle})")

    def _transition_to(self, phase: str) -> None:
        """Thread-safe phase transition; resets the phase timer."""
        with self.lock:
            self.phase = phase
            self.phase_start_time = time.time()
        print(f"[ParkingStrategy] Phase → {phase}")

    def _run_parking_sequence(self) -> None:
        """
        Background worker that drives the parking phase sequence at ~20 Hz.
        Transitions between phases based on elapsed time vs computed durations.
        """
        print("[ParkingStrategy] Worker thread started.")

        try:
            while True:
                with self.lock:
                    phase   = self.phase
                    start_t = self.phase_start_time
                    running = self.is_running

                if not running:
                    break

                elapsed = time.time() - start_t

                if phase == "WAIT":
                    # Autopilot is in control — no commands issued here.
                    if elapsed >= self.wait_duration:
                        if hasattr(self.controller, "autopilot_controller") and self.controller.autopilot_controller:
                            self.controller.autopilot_controller.pause()
                        self._transition_to("BRAKE1")

                elif phase == "BRAKE1":
                    self._set_motion(0, "forward", _SERVO_CENTER)
                    if elapsed >= self.brake_duration:
                        self._transition_to("FORWARD_LEFT")

                elif phase == "FORWARD_LEFT":
                    # Advance with full-left steering to angle the car toward the space
                    self._set_motion(self.maneuver_speed_ui, "forward", _SERVO_LEFT)
                    if elapsed >= self.forward_left_duration:
                        self._transition_to("BACK_RIGHT")

                elif phase == "BACK_RIGHT":
                    # Reverse with full-right steering to enter the space diagonally
                    self._set_motion(self.maneuver_speed_ui, "backward", _SERVO_RIGHT)
                    if elapsed >= self.back_right_duration:
                        self._transition_to("BACK_LEFT")

                elif phase == "BACK_LEFT":
                    # Reverse with full-left steering to straighten inside the space
                    self._set_motion(self.maneuver_speed_ui, "backward", _SERVO_LEFT)
                    if elapsed >= self.back_left_duration:
                        self._transition_to("STOP")

                elif phase == "STOP":
                    # Hold speed=0 for stop_hold_duration s so the ESP32 processes
                    # the stop command before this thread exits.
                    self._set_motion(0, "forward", _SERVO_CENTER)
                    if elapsed >= self.stop_hold_duration:
                        with self.lock:
                            self.is_running = False
                            self.is_parked  = True
                            self.phase      = "IDLE"

                        print("[ParkingStrategy] Maneuver complete. is_parked=True")

                        if self.controller.event_callback:
                            self.controller.event_callback("parking_completed", {"is_parked": True})
                        break

                else:
                    print(f"[ParkingStrategy] Unknown phase '{phase}' — aborting safely.")
                    self._set_motion(0, "forward", _SERVO_CENTER)
                    with self.lock:
                        self.is_running = False
                        self.phase      = "IDLE"
                    break

                time.sleep(0.05)  # ~20 Hz

        except Exception as exc:
            print(f"[ParkingStrategy] Unexpected error in worker thread: {exc}")
            try:
                self._set_motion(0, "forward", _SERVO_CENTER)
            except Exception:
                pass
            with self.lock:
                self.is_running = False
                self.phase      = "IDLE"
