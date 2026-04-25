"""
CR1000 Datalogger — Sensor Monitor (RS232) with CSV output
===========================================================
Connects to a Campbell Scientific CR1000 datalogger over RS232 serial,
continuously polls live sensor readings, prints them to terminal, and
appends each record to a CSV file.

Usage:
    python cr1000_monitor.py --port COM3 --baud 115200

    Full options:
    python cr1000_monitor.py --port COM3 --baud 115200 --table Table1
                             --interval 2.0 --pakbus 1
                             --output readings.csv

Arguments:
    --port      Serial port (required). Find in Device Manager -> Ports (COM & LPT)
    --baud      Baud rate. Must match CR1000 setting (default: 115200)
    --table     Data table name on the CR1000 (default: Table1)
    --interval  Seconds between readings (default: 2.0)
    --pakbus    CR1000 PakBus address (default: 1)
    --output    CSV output file path (default: cr1000_readings.csv)

Requirements:
    pip install pycampbellcr1000

Notes:
    - PC400 must be CLOSED before running — they cannot share the same COM port
    - CSV file is created if it does not exist, appended to if it does
    - Each row in the CSV is one sensor snapshot with a timestamp
    - Press Ctrl+C to stop monitoring cleanly
"""

import time
import argparse
import sys
import csv
import os
from datetime import datetime

try:
    from pycampbellcr1000 import CR1000
except ImportError:
    print("[ERROR] pycampbellcr1000 is not installed.")
    print("        Run: pip install pycampbellcr1000")
    sys.exit(1)


def log(level: str, message: str) -> None:
    """Print a timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    symbols = {
        "INFO":   "[INFO]   ",
        "OK":     "[OK]     ",
        "WARN":   "[WARN]   ",
        "ERROR":  "[ERROR]  ",
        "DATA":   "[DATA]   ",
        "ACTION": "[ACTION] ",
        "CSV":    "[CSV]    ",
    }
    prefix = symbols.get(level, "[LOG]    ")
    print(f"  {timestamp}  {prefix} {message}")


def connect(port: str, baud: int, pakbus: int) -> CR1000 | None:
    """Attempt to connect to the CR1000 over RS232 serial."""
    address = f"serial:{port}:{baud}"
    log("INFO", f"Connecting to CR1000 on {port} at {baud} baud ...")
    log("INFO", f"Protocol: PakBus over RS232  |  PakBus address: {pakbus}")

    try:
        device = CR1000.from_url(address, dest_addr=pakbus)
        log("OK", "Connection established successfully")
        return device
    except Exception as e:
        log("ERROR", f"Connection failed: {type(e).__name__}: {e}")
        log("ERROR", "Troubleshooting checklist:")
        log("ERROR", f"  1. Is the CR1000 powered ON?")
        log("ERROR", f"  2. Is the COM port correct? (currently: {port})")
        log("ERROR", f"  3. Does the baud rate match CR1000? (currently: {baud})")
        log("ERROR",  "  4. Is PC400 open and using this port? Close it first.")
        log("ERROR",  "  5. Is the RS232 cable firmly connected?")
        log("ERROR",  "  6. Does the CR1000 have an active measurement program?")
        return None


def get_tables(device: CR1000) -> list:
    """List all data tables available on the CR1000."""
    log("INFO", "Fetching available data tables from CR1000...")
    try:
        tables = device.list_tables()
        tables = [t.decode() if isinstance(t, bytes) else t for t in tables]
    except Exception as e:
        log("ERROR", f"Failed to list tables: {type(e).__name__}: {e}")
        return []


def get_fields(device: CR1000, table: str) -> list:
    """List all field names in a given table."""
    log("INFO", f"Fetching field names from table '{table}'...")
    try:
        fields = device.list_fields(table)
        log("OK", f"Fields in '{table}': {', '.join(fields)}")
        return fields
    except Exception as e:
        log("ERROR", f"Failed to list fields: {type(e).__name__}: {e}")
        return []


def fetch_latest(device: CR1000, table: str) -> dict | None:
    """Fetch the most recent record from the specified table."""
    try:
        records = device.get_data(table, start_date=None)

        if not records:
            log("WARN", f"No records found in table '{table}'")
            return None

        raw = records[-1]
        print("RAW RECORD:", raw)  # keep for debugging (can remove later)

        cleaned = {}

        for k, v in raw.items():
            # Fix keys like "b'PTemp_C'" → "PTemp_C"
            if isinstance(k, str) and k.startswith("b'") and k.endswith("'"):
                k = k[2:-1]

            # Optional: ensure Datetime stays clean
            if k == "Datetime":
                cleaned["Datetime"] = v
            else:
                cleaned[k] = v

        return cleaned

    except KeyboardInterrupt:
        raise
    except Exception as e:
        log("ERROR", f"Failed to fetch data: {type(e).__name__}: {e}")
        return None


def print_record(record: dict, fields: list) -> None:
    """Pretty-print a single sensor record to terminal."""
    print()
    print(f"  {'─' * 50}")
    timestamp = record.get("Datetime", "N/A")
    print(f"  {'Timestamp':<30} {timestamp}")
    for field in fields:
        if field == "Datetime":
            continue
        value = record.get(field, "N/A")
        if isinstance(value, float):
            print(f"  {field:<30} {value:.4f}")
        else:
            print(f"  {field:<30} {value}")
    print(f"  {'─' * 50}")


def init_csv(output_path: str, fields: list) -> bool:
    """
    Initialize the CSV file. Write header row if file does not exist.
    If file exists, append to it without rewriting the header.
    Returns True if ready to write, False on error.
    """
    file_exists = os.path.isfile(output_path)
    try:
        if not file_exists:
            log("CSV", f"Creating new CSV file: {output_path}")
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
            log("CSV", f"Header written: {', '.join(fields)}")
        else:
            log("CSV", f"Appending to existing CSV file: {output_path}")
            log("CSV", f"File size before start: {os.path.getsize(output_path)} bytes")
        return True
    except OSError as e:
        log("ERROR", f"Could not initialize CSV file '{output_path}': {e}")
        log("ERROR", "Check that the directory exists and you have write permissions")
        return False


def write_csv(output_path: str, record: dict, fields: list) -> bool:
    """
    Append a single record to the CSV file.
    Returns True on success, False on error.
    """
    try:
        with open(output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            row = {field: record.get(field, "") for field in fields}
            writer.writerow(row)
        return True
    except OSError as e:
        log("ERROR", f"Failed to write to CSV: {e}")
        return False


def test_connection(device: CR1000, table: str) -> bool:
    """Verify connection by listing tables and fetching one record."""
    log("INFO", "Running connection test...")

    log("INFO", f"Trying direct read from table '{table}' (bypassing list_tables)...")

    record = fetch_latest(device, table)
    if record is None:
        log("ERROR", "Could not fetch a record from table — communication issue or empty table")
        return False

    log("OK", "Direct read worked — table exists and data is accessible")
    return True

    # if table not in tables:
    #     log("WARN", f"Table '{table}' not found on device")
    #     log("WARN", f"Available tables: {', '.join(tables)}")
    #     log("WARN",  "Use --table to specify the correct table name from the list above")
    #     return False

    # log("INFO", f"Fetching one test record from table '{table}'...")
    # record = fetch_latest(device, table)
    # if record is None:
    #     log("ERROR", "Could not fetch a record — table may be empty")
    #     return False

    # log("OK", "Connection test PASSED — CR1000 is live and returning data")
    # return True


def monitor(device: CR1000, table: str, fields: list,
            interval: float, output_path: str) -> None:
    """
    Main polling loop — fetches latest record every interval seconds,
    prints to terminal and appends to CSV.
    """
    log("INFO", f"Starting live monitoring of table '{table}'")
    log("INFO", f"Poll interval: {interval}s")
    log("INFO", f"CSV output: {os.path.abspath(output_path)}")
    log("INFO", "Press Ctrl+C to stop")
    print()

    reading_count = 0
    error_count   = 0
    csv_error     = False
    last_timestamp = None
    fields_initialized = False

    # Optional: remove old CSV to avoid bad headers
    if os.path.exists(output_path):
        os.remove(output_path)
        log("CSV", f"Removed old CSV file: {output_path}")

    while True:
        log("ACTION", f"Fetching latest record from '{table}'...")
        record = fetch_latest(device, table)

        if record:
            ts = record.get("Datetime")

            # Skip duplicate timestamps
            if ts == last_timestamp:
                time.sleep(interval)
                continue

            last_timestamp = ts
            reading_count += 1
            error_count   = 0

            # 🔥 Initialize fields dynamically from first valid record
            if not fields_initialized:
                fields.clear()

                # Optional: exclude RecNbr if you don’t want it
                detected_fields = [k for k in record.keys() if k != "RecNbr"]

                # Ensure Datetime is first
                fields.extend(["Datetime"] + [f for f in detected_fields if f != "Datetime"])

                log("INFO", f"Detected fields: {', '.join(fields)}")

                # Initialize CSV with correct header
                if not init_csv(output_path, fields):
                    log("ERROR", "Failed to initialize CSV with detected fields")
                    return

                fields_initialized = True

            log("DATA", f"Record #{reading_count} received")
            print_record(record, fields)

            # Write to CSV
            if not csv_error:
                success = write_csv(output_path, record, fields)
                if success:
                    log("CSV", f"Record #{reading_count} written to {output_path}")
                else:
                    csv_error = True
                    log("WARN", "CSV writing disabled due to repeated errors")
                    log("WARN", "Data is still being printed to terminal")

        else:
            error_count += 1
            log("WARN", f"No data returned — failed reads: {error_count}")

            if error_count >= 5:
                log("ERROR", "5 consecutive failed reads — check CR1000 connection")
                log("ERROR", "Stopping monitor. Restart script to reconnect.")
                break

        try:
            time.sleep(interval)
        except KeyboardInterrupt:
            raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CR1000 datalogger live sensor monitor over RS232 with CSV output"
    )
    parser.add_argument(
        "--port",
        required=True,
        help="Serial port. Windows: COM3, COM4. Linux: /dev/ttyUSB0"
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Baud rate — must match CR1000 setting (default: 115200)"
    )
    parser.add_argument(
        "--table",
        default="Table1",
        help="Data table name on the CR1000 (default: Table1)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Poll interval in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--pakbus",
        type=int,
        default=1,
        help="CR1000 PakBus address (default: 1)"
    )
    parser.add_argument(
        "--output",
        default="cr1000_readings.csv",
        help="CSV output file path (default: cr1000_readings.csv)"
    )
    args = parser.parse_args()

    print()
    print("=" * 55)
    print("  CR1000 Datalogger — Sensor Monitor (RS232)")
    print("=" * 55)
    log("INFO", f"Port: {args.port}  |  Baud: {args.baud}  |  "
               f"Table: {args.table}  |  PakBus: {args.pakbus}")
    log("INFO", f"Poll interval: {args.interval}s")
    log("INFO", f"Output CSV: {os.path.abspath(args.output)}")
    log("WARN", "Make sure PC400 is CLOSED before running this script")

    # Connect
    device = connect(args.port, args.baud, args.pakbus)
    if device is None:
        log("ERROR", "Exiting due to failed connection")
        sys.exit(1)

    # Test connection
    if not test_connection(device, args.table):
        log("ERROR", "Exiting due to failed connection test")
        sys.exit(1)

    # Get fields
    fields = get_fields(device, args.table)
    if not fields:
        log("WARN", "Could not retrieve field names — will print raw records")
        fields = ["Datetime"]

    # Initialize CSV
    if not init_csv(args.output, fields):
        log("ERROR", "Exiting due to CSV initialization failure")
        sys.exit(1)

    print()

    # Start monitoring
    try:
        monitor(device, args.table, fields, args.interval, args.output)
    except KeyboardInterrupt:
        print()
        log("INFO", "Monitoring stopped by user (Ctrl+C)")
        log("INFO", f"CSV saved to: {os.path.abspath(args.output)}")
        log("OK",   "Goodbye.")
        sys.exit(0)


if __name__ == "__main__":
    main()