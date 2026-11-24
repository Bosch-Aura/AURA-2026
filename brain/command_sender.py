"""
Command Sender Module - Sends commands to the low-level controller via UART.

Two low-level controllers are supported, selected with set_protocol() (or the
AURA_UART_PROTOCOL environment variable, or --protocol in dashboard_server.py):

  "esp32"  (default) - ESP32 FreeRTOS firmware (repo "embedded").
           Format: CHANNEL:COMMAND[:VALUE]\\n
             C:SET_SPEED:<0-255>   C:SET_STEER:<50-160>
             E:BRAKE_NOW           M:SYS_ARM / M:SYS_DISARM / M:SYS_MODE:<0|1>
             M:LIGHTS_ON / M:LIGHTS_OFF / M:LIGHTS_AUTO
           Any parseable line refreshes the ESP32 watchdog (heartbeat).
           The ESP32 has NO reverse: SET_SPEED is cast to uint8, so a negative
           value would become a FORWARD speed. Backward requests are refused.

  "nucleo" - STM32 Nucleo firmware. Format: #key:payload;;\\r\\n
           Keys (lowercase, from main.cpp g_serialMonitorSubscribers):
           speed (mm/s), steer (tenths of degree), brake, alive, kl.
"""

import os

PROTOCOL_ESP32 = "esp32"
PROTOCOL_NUCLEO = "nucleo"
SUPPORTED_PROTOCOLS = (PROTOCOL_ESP32, PROTOCOL_NUCLEO)

_protocol = os.environ.get("AURA_UART_PROTOCOL", PROTOCOL_ESP32).strip().lower()
if _protocol not in SUPPORTED_PROTOCOLS:
    print(f"[CommandSender] Unknown AURA_UART_PROTOCOL '{_protocol}', using '{PROTOCOL_ESP32}'")
    _protocol = PROTOCOL_ESP32


def set_protocol(protocol: str) -> None:
    """Select the UART protocol ("esp32" or "nucleo") for every CommandSender."""
    global _protocol
    protocol = protocol.strip().lower()
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported protocol '{protocol}'. Use one of {SUPPORTED_PROTOCOLS}")
    _protocol = protocol


def get_protocol() -> str:
    return _protocol


# Servo / speed ranges (UI units shared by both protocols)
SERVO_CENTER = 105
SERVO_RIGHT = 50
SERVO_LEFT = 160
ANGLE_MAX = 30
SPEED_UI_MAX = 255

# Nucleo-specific ranges
STEER_TENTHS_MIN = -250
STEER_TENTHS_MAX = 250
SPEED_MM_S_MIN = -500
SPEED_MM_S_MAX = 500
UART_TERMINATOR = ";;\r\n"

# Nucleo serial keys (from main.cpp g_serialMonitorSubscribers - lowercase)
KEY_STEER = "steer"
KEY_SPEED = "speed"
KEY_BRAKE = "brake"
KEY_ALIVE = "alive"
KEY_KL = "kl"


def _nucleo(key: str, payload) -> str:
    return f"#{key}:{payload}{UART_TERMINATOR}"


def _servo_to_steer_tenths(servo_angle: int) -> int:
    conversion_factor = (SERVO_CENTER - SERVO_RIGHT) / ANGLE_MAX
    steering_angle = (SERVO_CENTER - servo_angle) / conversion_factor
    steering_tenths = int(round(steering_angle * 10))
    return max(STEER_TENTHS_MIN, min(STEER_TENTHS_MAX, steering_tenths))


def _ui_speed_to_mm_s(speed: int, direction: str) -> int:
    scaled = int(round((speed / SPEED_UI_MAX) * SPEED_MM_S_MAX))
    scaled = -abs(scaled) if direction.lower() == "backward" else abs(scaled)
    return max(SPEED_MM_S_MIN, min(SPEED_MM_S_MAX, scaled))


# ---------------------------------------------------------------------------
# Command builders: return the UART line for the active protocol, or None if
# the active controller does not support that command.
# ---------------------------------------------------------------------------

def build_steer(servo_angle: int) -> str:
    """servo_angle: 50-160, 105 = center."""
    servo_angle = max(SERVO_RIGHT, min(SERVO_LEFT, int(servo_angle)))
    if _protocol == PROTOCOL_ESP32:
        return f"C:SET_STEER:{servo_angle}"
    return _nucleo(KEY_STEER, _servo_to_steer_tenths(servo_angle))


def build_speed(speed: int, direction: str = "forward"):
    """speed: 0-255 (UI units). Returns None for backward on the ESP32."""
    speed = max(0, min(SPEED_UI_MAX, int(speed)))
    if _protocol == PROTOCOL_ESP32:
        if direction.lower() == "backward" and speed > 0:
            return None
        return f"C:SET_SPEED:{speed}"
    return _nucleo(KEY_SPEED, _ui_speed_to_mm_s(speed, direction))


def build_brake() -> str:
    if _protocol == PROTOCOL_ESP32:
        # Note: the ESP32 ignores SET_SPEED for 5 s after a brake (cooldown).
        return "E:BRAKE_NOW"
    return _nucleo(KEY_BRAKE, 0)


def build_heartbeat() -> str:
    if _protocol == PROTOCOL_ESP32:
        # The ESP32 has no HEARTBEAT command, but every parseable line refreshes
        # its watchdog and an unknown M: command is silently ignored.
        return "M:HEARTBEAT:1"
    return _nucleo(KEY_ALIVE, 1)


def build_arm() -> str:
    return "M:SYS_ARM" if _protocol == PROTOCOL_ESP32 else _nucleo(KEY_KL, 30)


def build_disarm() -> str:
    return "M:SYS_DISARM" if _protocol == PROTOCOL_ESP32 else _nucleo(KEY_KL, 0)


def build_mode(auto: bool) -> str:
    if _protocol == PROTOCOL_ESP32:
        return f"M:SYS_MODE:{1 if auto else 0}"
    return _nucleo(KEY_KL, 30 if auto else 15)


def build_lights(state: str):
    """state: "on", "off" or "auto". Returns None on the Nucleo (unsupported)."""
    if _protocol != PROTOCOL_ESP32:
        return None
    return {"on": "M:LIGHTS_ON", "off": "M:LIGHTS_OFF", "auto": "M:LIGHTS_AUTO"}.get(state.lower())


class CommandSender:
    """Sends steering, speed, brake and heartbeat to the low-level controller."""

    def __init__(self, write_uart_command_func):
        """
        Args:
            write_uart_command_func: Function that takes a UART command string
                                     and returns (success: bool, message: str)
        """
        self.write_uart_command = write_uart_command_func

    def _send(self, command) -> bool:
        if command is None:
            return False
        success, _message = self.write_uart_command(command)
        return success

    def send_steering_command(self, servo_angle: int) -> bool:
        """servo_angle: 50-160, where 105 is center."""
        return self._send(build_steer(servo_angle))

    def send_speed_command(self, speed: int, direction: str = "forward") -> bool:
        """
        speed: 0-255. direction: "forward" or "backward".

        On the ESP32 backward is not supported: the car is stopped instead and
        False is returned, so callers do not think they are reversing.
        """
        command = build_speed(speed, direction)
        if command is None:
            print("[CommandSender] ESP32 has no reverse - sending speed 0 instead of backward")
            self._send(build_speed(0))
            return False
        return self._send(command)

    def send_brake(self) -> bool:
        return self._send(build_brake())

    def send_heartbeat(self) -> bool:
        return self._send(build_heartbeat())

    def send_arm(self) -> bool:
        return self._send(build_arm())

    def send_disarm(self) -> bool:
        return self._send(build_disarm())

    def send_mode(self, auto: bool) -> bool:
        return self._send(build_mode(auto))
