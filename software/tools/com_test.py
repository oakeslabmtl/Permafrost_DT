import serial, time

s = serial.Serial('COM4', 9600, timeout=5)
print("Listening on COM4 for 10s...")
time.sleep(10)
data = s.read(1000)
print(f"Received {len(data)} bytes: {repr(data)}")
s.close()