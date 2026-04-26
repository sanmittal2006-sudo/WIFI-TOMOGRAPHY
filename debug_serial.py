import serial
import time

PORT = 'COM7'
BAUD = 115200

print(f"Opening {PORT} at {BAUD} baud...")
s = serial.Serial(PORT, BAUD, timeout=2)
time.sleep(3)  # Wait for ESP32 to boot

print("Reading raw bytes for 10 seconds...\n")

# First, read raw bytes to see what's coming
raw = s.read(2000)
print(f"Got {len(raw)} raw bytes")
print(f"First 500 bytes:")
print(raw[:500])
print(f"\n--- Decoded ---")
try:
    text = raw.decode('utf-8', errors='ignore')
    print(text[:500])
except:
    print("Cannot decode")

# Now try readline
print(f"\n--- Reading 10 lines ---")
for i in range(10):
    line = s.readline()
    decoded = line.decode('utf-8', errors='ignore').strip()
    print(f"Line {i}: len={len(line)}, has CSI={'CSI_DATA' in decoded}, content={decoded[:100]}")

s.close()
print("\nDone!")
