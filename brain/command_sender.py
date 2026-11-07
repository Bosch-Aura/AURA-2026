"""
Command Sender Module - Sends commands to the ESP32 (Bosch-Aura/embedded) via UART.

Protocol (embedded/docs/BRAIN_TEAM_PROTOCOL.md): CHANNEL:COMMAND:VALUE\n
    C = control (SET_SPEED, SET_STEER, SET_DIR)
    E = emergency (BRAKE_NOW)
    M = management (PING, SYS_ARM, SYS_DISARM, OBSTACLE_RESET)

Speed and steering are *latching* on the ESP32: one command holds until the next
one, so the brain sends a speed once and only again when it changes. The link
watchdog (300 ms) is fed with M:PING:0 from the dashboard heartbeat.

On an obstacle the ESP32 brakes, keeps the speed (EVENT:OBSTACLE_HOLD:<pwm>) and
resumes it by itself when the path clears (EVENT:OBSTACLE_RESUMED:<pwm>). The
brain does not re-send anything for that.
"""

import threading


# Servo values sent as-is to C:SET_STEER (firmware clamps to 50-160).
# NOTE: Brain convention is 50 = right, 160 = left. The firmware docs and the
# ROS2 bridge label 50 as LEFT. If the car steers the wrong way, set
# STEER_INVERTED = True: commands are mirrored around SERVO_CENTER here and the
# rest of the brain keeps its convention.
SERVO_CENTER = 105
SERVO_MIN = 50
SERVO_MAX = 160
STEER_INVERTED = False

# C:SET_SPEED is a PWM duty (0-255). The firmware's start threshold is 130
# (bench, 2026-11-06): below it the motor hums but does not turn, and the
# firmware still reports CMD_EXECUTED, so the failure is silent. DUTY_START
# keeps a margin for the loaded car on the floor; any non-zero UI speed (1-255)
# is mapped onto DUTY_START..DUTY_MAX.
SPEED_UI_MAX = 255
DUTY_START = 140
DUTY_MAX = 255

CMD_TERMINATOR = "\n"


def format_command(channel: str, command: str, value: int = 0) -> str:
    """Build a protocol line: CHANNEL:COMMAND:VALUE\\n"""
    return f"{channel}:{command}:{int(value)}{CMD_TERMINATOR}"


def ui_speed_to_duty(speed: int) -> int:
    """Map UI speed (0-255) to ESP32 duty: 0 stays 0, anything else starts at DUTY_START."""
    speed = max(0, min(SPEED_UI_MAX, int(speed)))
    if speed == 0:
        return 0
    return int(round(DUTY_START + (speed / SPEED_UI_MAX) * (DUTY_MAX - DUTY_START)))


def servo_to_firmware(servo_angle: int) -> int:
    """Clamp a brain servo value and apply STEER_INVERTED."""
    servo_angle = max(SERVO_MIN, min(SERVO_MAX, int(servo_angle)))
    if STEER_INVERTED:
        servo_angle = 2 * SERVO_CENTER - servo_angle
    return servo_angle


class CommandSender:
    """Sends steering, speed, brake, arming and heartbeat to the ESP32 over UART."""

    def __init__(self, write_uart_command_func):
        """
        Initialize the command sender.

        Args:
            write_uart_command_func: Function that takes a UART command string
                                     and returns (success: bool, message: str)
        """
        self.write_uart_command = write_uart_command_func
        self.lock = threading.Lock()
        # Last speed/direction requested by the brain (UI scale 0-255).
        self.last_speed = 0
        self.last_direction = "forward"
        # Direction the ESP32 currently has; None = unknown (forces C:SET_DIR).
        self._firmware_forward = None
        # Duty the ESP32 has latched, or keeps to resume after an obstacle.
        # None = unknown (forces C:SET_SPEED). Updated optimistically on send and
        # corrected by the ESP32 events (see on_firmware_speed).
        self._firmware_duty = None

    def _send(self, command: str) -> bool:
        success, _message = self.write_uart_command(command)
        return success

    def reset_link_state(self):
        """Forget what the ESP32 has latched. Call after the board (re)boots."""
        with self.lock:
            self._firmware_forward = None
            self._firmware_duty = None
            self.last_speed = 0

    def forget_speed(self):
        """
        The ESP32 dropped its latched speed for good (DISARMED, FAULT, SYS_STOP,
        rejected SET_SPEED, OBSTACLE_RESET, reboot). The next speed is always sent.
        """
        with self.lock:
            self.last_speed = 0
            self._firmware_duty = None

    def on_firmware_speed(self, duty: int):
        """
        The ESP32 reported the duty it has latched (CMD_EXECUTED:SET_SPEED) or keeps
        to resume (OBSTACLE_HOLD / OBSTACLE_RESUMED). It is the source of truth.
        """
        with self.lock:
            self._firmware_duty = int(duty)

    def send_steering_command(self, servo_angle: int) -> bool:
        """
        Send steering command (C:SET_STEER).

        Args:
            servo_angle: Servo angle (50-160, where 105 is center)

        Returns:
            True if command was sent successfully, False otherwise
        """
        return self._send(format_command("C", "SET_STEER", servo_to_firmware(servo_angle)))

    def send_speed_command(self, speed: int, direction: str = "forward") -> bool:
        """
        Set the speed once: C:SET_DIR when the direction changes, then C:SET_SPEED.

        The ESP32 latches it, so nothing is sent when it already has this duty and
        direction (also while it holds it for an obstacle). A stop (0) always goes out.

        Args:
            speed: Speed value (0-255)
            direction: Direction ("forward" or "backward")

        Returns:
            True if the command was sent or was already latched, False otherwise
        """
        speed = max(0, min(SPEED_UI_MAX, int(speed)))
        forward = direction.lower() != "backward"
        duty = ui_speed_to_duty(speed)

        with self.lock:
            self.last_speed = speed
            self.last_direction = "forward" if forward else "backward"

            if speed > 0 and duty == self._firmware_duty and self._firmware_forward == forward:
                return True

            if speed > 0 and self._firmware_forward != forward:
                if not self._send(format_command("C", "SET_DIR", 1 if forward else 0)):
                    return False
                self._firmware_forward = forward
                # SET_DIR during an obstacle hold drops the kept speed.
                self._firmware_duty = None

            if not self._send(format_command("C", "SET_SPEED", duty)):
                self._firmware_duty = None
                return False
            self._firmware_duty = duty
            return True

    def send_brake(self) -> bool:
        """
        Emergency brake (E:BRAKE_NOW). One-shot: the ESP32 brakes and drops the latched
        speed, and the next C:SET_SPEED moves the car again (no lock-out).

        Returns:
            True if command was sent successfully, False otherwise
        """
        with self.lock:
            self.last_speed = 0
            self._firmware_duty = 0
        return self._send(format_command("E", "BRAKE_NOW"))

    def send_arm(self) -> bool:
        """Arm the ESP32 (M:SYS_ARM). Required after every boot and to leave FAULT."""
        return self._send(format_command("M", "SYS_ARM"))

    def send_disarm(self) -> bool:
        """Disarm the ESP32 (M:SYS_DISARM): brakes, centers steering, clears latched speed."""
        with self.lock:
            self.last_speed = 0
            self._firmware_duty = 0
        return self._send(format_command("M", "SYS_DISARM"))

    def send_obstacle_reset(self) -> bool:
        """
        Release the ESP32 ultrasonic hold (M:OBSTACLE_RESET).

        The ESP32 also clears on its own (valid far readings, or 2 s with no echo).
        A manual reset drops the kept speed: the car does NOT resume by itself,
        it needs a new speed command. If the obstacle is still there it blocks again.
        """
        with self.lock:
            self._firmware_duty = None
        return self._send(format_command("M", "OBSTACLE_RESET"))

    def send_heartbeat(self) -> bool:
        """
        Send heartbeat (M:PING) so the ESP32 link watchdog (300 ms) does not trip.

        Returns:
            True if command was sent successfully, False otherwise
        """
        return self._send(format_command("M", "PING"))
