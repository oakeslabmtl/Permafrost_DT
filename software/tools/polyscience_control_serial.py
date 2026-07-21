"""
PolyScience Circulator — Heater Control (RS232/Serial)
======================================================
Interactive menu to control a PolyScience Advanced Digital Temperature
Controller over RS232 serial connection (direct or via USB adapter).

Usage:
    python polyscience_control_serial.py --port COM3 --baud 57600

Setup:
    1. Connect RS232 cable (or RS232-to-USB adapter) from PC to controller DB9 port
    2. On the controller: press menu key until COM RS232 appears, select it
    3. Set baud rate on controller to match --baud argument (default 57600)
    4. Find your COM port:
       - Windows: Device Manager -> Ports (COM & LPT)

Serial wiring (DB9):
    Pin 2 -- TX
    Pin 3 -- RX
    Pin 5 -- GND

"""

import time
import argparse
import sys
from datetime import datetime

try:
    import serial
except ImportError:
    print("[ERROR] pyserial is not installed.")
    print("        Run: pip install pyserial")
    sys.exit(1)


DEFAULT_PORT    = "COM3"
DEFAULT_BAUD    = 57600
DEFAULT_TIMEOUT = 3.0
SERIAL_SETTINGS = {
    "bytesize": serial.EIGHTBITS,
    "parity":   serial.PARITY_NONE,
    "stopbits": serial.STOPBITS_ONE,
    "xonxoff":  False,
    "rtscts":   False,
}


def log(level: str, message: str) -> None:
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    symbols = {
        "INFO":   "[INFO]   ",
        "OK":     "[OK]     ",
        "WARN":   "[WARN]   ",
        "ERROR":  "[ERROR]  ",
        "SEND":   "[SEND]   ",
        "RECV":   "[RECV]   ",
        "ACTION": "[ACTION] ",
    }
    prefix = symbols.get(level, "[LOG]    ")
    print(f"  {timestamp}  {prefix} {message}")


def send_command(ser: serial.Serial, command: str) -> str:
    """
    Send a single RS232 command and return the response.
    Commands end with carriage return [CR].
    '!' prefix = success, '?' prefix = error.
    """
    try:
        ser.reset_input_buffer()
        raw_cmd = (command + "\r").encode("ascii")
        log("SEND", f"Command: '{command}\\r'  ({len(raw_cmd)} bytes)")
        ser.write(raw_cmd)

        response_bytes = ser.read_until(b"\r", size=64)

        if not response_bytes:
            log("ERROR", "No response received — connection timed out waiting for [CR]")
            log("ERROR", "Possible causes: wrong baud rate, cable unplugged, controller not in RS232 mode")
            return "TIMEOUT"

        response = response_bytes.decode("ascii").strip()
        log("RECV", f"Response: '{response}'  ({len(response_bytes)} bytes)")

        if response.startswith("!"):
            log("OK", "Controller acknowledged command successfully")
        elif response.startswith("?"):
            log("WARN", "Controller returned '?' — command rejected or value out of range")
            log("WARN", "Possible causes: command syntax error, setpoint out of allowed range, controller locked")
        else:
            log("WARN", f"Unexpected response format: '{response}' — expected '!' or '?'")

        return response

    except serial.SerialTimeoutException:
        log("ERROR", "Serial write timed out — controller not responding")
        return "TIMEOUT"
    except serial.SerialException as e:
        log("ERROR", f"Serial communication error: {e}")
        return f"ERROR: {e}"
    except UnicodeDecodeError as e:
        log("ERROR", f"Failed to decode response as ASCII: {e}")
        log("ERROR", "Possible cause: baud rate mismatch between script and controller")
        return "DECODE_ERROR"
    except Exception as e:
        log("ERROR", f"Unexpected error during command: {type(e).__name__}: {e}")
        return f"ERROR: {e}"


def is_success(response: str) -> bool:
    return response.startswith("!")


def read_state(ser: serial.Serial) -> dict:
    """Read current temperature, setpoint, operating status, and alarm state."""
    log("INFO", "Reading full controller state (RT, RS, RO, RF)...")

    raw_temp   = send_command(ser, "RT")
    raw_setpt  = send_command(ser, "RS")
    raw_status = send_command(ser, "RO")
    raw_alarm  = send_command(ser, "RF")

    def parse_float(r, label):
        if r in ("TIMEOUT", "DECODE_ERROR") or r.startswith("ERROR"):
            log("WARN", f"Could not parse {label} — raw value: '{r}'")
            return r
        try:
            return f"{float(r[1:]):.2f} C" if r.startswith("!") else r
        except ValueError:
            log("WARN", f"Could not convert {label} value to float: '{r}'")
            return r

    status = (
        "Running" if raw_status == "!1" else
        "Standby" if raw_status == "!0" else
        raw_status
    )
    alarms = (
        "OK"    if raw_alarm == "!0" else
        "FAULT" if raw_alarm == "!1" else
        raw_alarm
    )

    if alarms == "FAULT":
        log("WARN", "Active fault detected on controller — check controller display")
    if status == "Running":
        log("INFO", "Heater is currently RUNNING")
    else:
        log("INFO", f"Heater status: {status}")

    return {
        "internal_temp": parse_float(raw_temp,  "internal temperature"),
        "setpoint":      parse_float(raw_setpt, "setpoint"),
        "status":        status,
        "alarms":        alarms,
    }


def print_state(state: dict) -> None:
    print()
    print(f"  {'─' * 38}")
    print(f"  {'Internal temp':<20} {state['internal_temp']}")
    print(f"  {'Setpoint':<20} {state['setpoint']}")
    print(f"  {'Status':<20} {state['status']}")
    print(f"  {'Alarms':<20} {state['alarms']}")
    print(f"  {'─' * 38}")


def print_menu() -> None:
    print("\n" + "=" * 45)
    print("  PolyScience Heater Control (RS232)")
    print("=" * 45)
    print("  1. Read current state")
    print("  2. Turn heater ON")
    print("  3. Turn heater OFF")
    print("  4. Set temperature setpoint")
    print("  5. Read setpoint")
    print("  6. Read alarm status")
    print("  0. Exit")
    print("=" * 45)


def confirm(prompt: str) -> bool:
    answer = input(f"\n  {prompt} [y/N]: ").strip().lower()
    confirmed = answer == "y"
    if not confirmed:
        log("INFO", "User cancelled action")
    return confirmed


def run_menu(ser: serial.Serial) -> None:
    """Main interactive control loop."""
    log("INFO", f"Session started on {ser.port} at {ser.baudrate} baud")
    log("INFO", f"Serial settings: {ser.bytesize} data bits, parity={ser.parity}, stopbits={ser.stopbits}")

    while True:
        print_menu()
        choice = input("  Enter choice: ").strip()
        print()

        if choice == "1":
            log("ACTION", "User selected: Read current state")
            state = read_state(ser)
            print_state(state)

        elif choice == "2":
            log("ACTION", "User selected: Turn heater ON")
            log("INFO", "Reading state before sending ON command...")
            state = read_state(ser)
            print_state(state)

            if state["alarms"] == "FAULT":
                log("WARN", "Heater ON command blocked — active fault must be resolved first")
                log("WARN", "Check the controller display for the fault code")
                continue


            log("INFO", "Sending heater ON command: SO1")
            response = send_command(ser, "SO1")
            if is_success(response):
                log("OK", "Heater turned ON successfully")
            else:
                log("ERROR", f"Heater ON command failed — response: '{response}'")
                log("ERROR", "Check that controller is powered and in RS232 mode")

        elif choice == "3":
            log("ACTION", "User selected: Turn heater OFF")

            log("INFO", "Sending heater OFF command: SO0")
            response = send_command(ser, "SO0")
            if is_success(response):
                log("OK", "Heater turned OFF successfully")
            else:
                log("ERROR", f"Heater OFF command failed — response: '{response}'")

        elif choice == "4":
            log("ACTION", "User selected: Set temperature setpoint")
            raw = input("  Enter new setpoint in C (e.g. 25.00): ").strip()

            try:
                value = float(raw)
            except ValueError:
                log("ERROR", f"Invalid input '{raw}' — could not convert to a number")
                continue

            if not (-30 <= value <= 50):
                log("ERROR", f"Setpoint {value:.2f} C is out of allowed range (0-200 C)")
                continue

            cmd = f"SS{value:06.2f}"
            log("INFO", f"Sending setpoint command: {cmd}")
            response = send_command(ser, cmd)
            if is_success(response):
                log("OK", f"Setpoint successfully set to {value:.2f} C")
            else:
                log("ERROR", f"Setpoint command failed — response: '{response}'")
                log("ERROR", "Check that the value is within the controller's allowed range")

        elif choice == "5":
            log("ACTION", "User selected: Read setpoint")
            log("INFO", "Sending read setpoint command: RS")
            response = send_command(ser, "RS")
            if response.startswith("!"):
                try:
                    setpoint = float(response[1:])
                    log("OK", f"Current setpoint: {setpoint:.2f} C")
                except ValueError:
                    log("WARN", f"Could not parse setpoint value from response: '{response}'")
            else:
                log("ERROR", f"Failed to read setpoint — response: '{response}'")

        elif choice == "6":
            log("ACTION", "User selected: Read alarm status")
            log("INFO", "Sending read alarm command: RF")
            response = send_command(ser, "RF")
            if response == "!0":
                log("OK", "Alarm status: No faults detected")
            elif response == "!1":
                log("WARN", "Alarm status: FAULT active — check controller display for fault code")
            else:
                log("ERROR", f"Unexpected alarm response: '{response}'")

        elif choice == "0":
            log("ACTION", "User selected: Exit")
            log("INFO", "Checking heater state before closing connection...")
            state = read_state(ser)
            if state["status"] == "Running":
                log("WARN", "Heater is still RUNNING — recommended to turn it OFF before exiting")
                if confirm("Turn heater OFF before exiting?"):
                    log("INFO", "Sending heater OFF command: SO0")
                    response = send_command(ser, "SO0")
                    if is_success(response):
                        log("OK", "Heater turned OFF")
                    else:
                        log("ERROR", f"Failed to turn heater OFF — response: '{response}'")
                        log("WARN", "Turn heater OFF manually on the controller before leaving")
            else:
                log("INFO", "Heater is not running — safe to exit")

            log("INFO", "Closing serial connection...")
            log("OK", "Session ended. Goodbye.")
            break

        else:
            log("WARN", f"Invalid choice '{choice}' — please enter a number shown in the menu")

        time.sleep(0.2)


def test_connection(ser: serial.Serial) -> bool:
    """Send a single read command to verify the connection is working."""
    log("INFO", f"Opening serial port {ser.port}...")
    log("INFO", f"Settings: {ser.baudrate} baud, {ser.bytesize} data bits, "
               f"parity={ser.parity}, stopbits={ser.stopbits}")
    log("INFO", "Sending test command RT (read internal temperature)...")

    response = send_command(ser, "RT")

    if response == "TIMEOUT":
        log("ERROR", "Connection test FAILED — no response received")
        log("ERROR", "Troubleshooting checklist:")
        log("ERROR", "  1. Is the RS232 cable physically connected?")
        log("ERROR", "  2. Is the controller powered ON?")
        log("ERROR", "  3. Is the controller COM mode set to RS232?")
        log("ERROR", f"  4. Is the COM port correct? (currently: {ser.port})")
        log("ERROR", f"  5. Does the baud rate match the controller? (currently: {ser.baudrate})")
        log("ERROR", "  6. Is PuTTY or another program already using this port?")
        return False

    if response.startswith("?"):
        log("WARN", f"Controller responded but returned error: '{response}'")
        log("WARN", "Connection is live but command was rejected — check controller state")
        return False

    if response.startswith("ERROR") or response == "DECODE_ERROR":
        log("ERROR", f"Communication error during connection test: '{response}'")
        log("ERROR", "Most likely cause: baud rate mismatch between script and controller")
        return False

    try:
        temp = float(response[1:])
        log("OK", f"Connection test PASSED — controller temperature: {temp:.2f} C")
    except ValueError:
        log("OK", f"Connection test PASSED — controller responded: '{response}'")

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PolyScience circulator interactive heater control via RS232"
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"Serial port (default: {DEFAULT_PORT}). Linux: /dev/ttyUSB0"
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Baud rate — must match controller setting (default: {DEFAULT_BAUD})"
    )
    args = parser.parse_args()

    print()
    print("=" * 55)
    print("  PolyScience Circulator — Heater Control (RS232)")
    print("=" * 55)
    log("INFO", f"Port: {args.port}  |  Baud: {args.baud}")
    log("INFO", "Attempting to open serial connection...")

    try:
        ser = serial.Serial(
            port=args.port,
            baudrate=args.baud,
            timeout=DEFAULT_TIMEOUT,
            **SERIAL_SETTINGS
        )
    except serial.SerialException as e:
        log("ERROR", f"Could not open port {args.port}")
        log("ERROR", f"System error: {e}")
        log("ERROR", "Possible causes:")
        log("ERROR", "  - Port does not exist (check Device Manager)")
        log("ERROR", "  - Port is already in use by PuTTY or another program")
        log("ERROR", "  - USB-to-RS232 adapter driver not installed")
        sys.exit(1)

    log("OK", f"Serial port {args.port} opened successfully")

    with ser:
        if not test_connection(ser):
            log("ERROR", "Exiting due to failed connection test")
            sys.exit(1)

        print()
        try:
            run_menu(ser)
        except KeyboardInterrupt:
            print()
            log("WARN", "Interrupted by user (Ctrl+C)")
            log("INFO", "Checking heater state before emergency exit...")
            state = read_state(ser)
            if state["status"] == "Running":
                log("WARN", "Heater is still RUNNING — turn it OFF manually on the controller")
            else:
                log("INFO", "Heater is not running — safe to exit")
            log("INFO", "Closing connection...")
            sys.exit(0)


if __name__ == "__main__":
    main()