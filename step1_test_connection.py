#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║  STEP 1: Test ESP32 Connection                       ║
║  Run this FIRST to verify your ESP32s are working    ║
╚══════════════════════════════════════════════════════╝

Usage:
    python step1_test_connection.py

What it does:
    1. Lists all COM ports on your PC
    2. Connects to the RX ESP32
    3. Reads 20 CSI packets
    4. Shows you the data in human-readable format
    5. Tells you if everything is working
"""

import serial
import serial.tools.list_ports
import time
import sys

# ═══════════════════════════════════════════════════════
#   CHANGE THIS TO YOUR COM PORT
# ═══════════════════════════════════════════════════════
RX_PORT   = 'COM7'       # Check Device Manager for your port
BAUD_RATE = 115200       # ESP-IDF default baud rate
# ═══════════════════════════════════════════════════════


def list_all_ports():
    """Show all available COM ports."""
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("  ❌ No COM ports found!")
        print("     → Is your ESP32 plugged in via USB?")
        return False
    print("  Available COM ports:")
    for p in ports:
        print(f"    {p.device:8s} — {p.description}")
    return True


def test_connection():
    """Try to connect and read CSI data."""
    
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Wi-Fi Tomography — Connection Test                  ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Step 1: List ports
    print("── STEP 1: Checking COM ports ──")
    if not list_all_ports():
        return False
    print()

    # Step 2: Connect to RX
    print(f"── STEP 2: Connecting to RX on {RX_PORT} ──")
    try:
        ser = serial.Serial(RX_PORT, BAUD_RATE, timeout=2)
        time.sleep(2)  # Wait for connection to stabilize
        print(f"  ✅ Connected to {RX_PORT} at {BAUD_RATE} baud")
    except serial.SerialException as e:
        print(f"  ❌ Cannot connect to {RX_PORT}")
        print(f"     Error: {e}")
        print()
        print("  TROUBLESHOOTING:")
        print("    1. Check the COM port number in Device Manager")
        print(f"    2. Change RX_PORT in this script to match")
        print("    3. Close any Serial Monitor windows in Arduino IDE")
        print("    4. Make sure USB cable is a DATA cable (not charge-only)")
        return False
    print()

    # Step 3: Read raw data
    print("── STEP 3: Reading serial data (10 seconds) ──")
    print("  Looking for CSI_DATA packets...")
    print()
    
    csi_count = 0
    other_count = 0
    start = time.time()
    
    # Clear old buffer
    ser.reset_input_buffer()
    
    while time.time() - start < 10:
        try:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode('utf-8', errors='ignore').strip()
            
            if not line:
                continue
            
            if line.startswith('CSI_DATA'):
                csi_count += 1
                
                # Parse and display first 5 packets in detail
                if csi_count <= 5:
                    parts = line.split(',', 5)
                    if len(parts) >= 5:
                        seq   = parts[1]
                        rssi  = parts[2]
                        noise = parts[3]
                        n_val = parts[4]
                        
                        # Count subcarrier pairs
                        bracket_start = line.find('[')
                        bracket_end   = line.find(']')
                        if bracket_start >= 0 and bracket_end >= 0:
                            vals = line[bracket_start+1:bracket_end].split(',')
                            n_subcarriers = len(vals) // 2
                        else:
                            n_subcarriers = '?'
                        
                        print(f"  📡 Packet #{seq}")
                        print(f"     RSSI:         {rssi} dBm")
                        print(f"     Noise Floor:  {noise} dBm")
                        print(f"     Raw Values:   {n_val}")
                        print(f"     Subcarriers:  {n_subcarriers}")
                        print()
                
                elif csi_count == 6:
                    print("  ... (showing remaining packets as dots)")
                    print("  ", end="", flush=True)
                else:
                    print(".", end="", flush=True)
                    
            elif line.startswith('#'):
                # Status/heartbeat message
                other_count += 1
                if other_count <= 3:
                    print(f"  ℹ️  {line}")
            else:
                other_count += 1
                if other_count <= 3:
                    print(f"  📝 {line}")
                    
        except Exception as e:
            continue
    
    print()
    print()
    ser.close()

    # Step 4: Results
    print("── STEP 4: Results ──")
    print()
    
    if csi_count >= 10:
        print(f"  ✅ SUCCESS! Received {csi_count} CSI packets in 10 seconds")
        print(f"     Rate: ~{csi_count/10:.0f} packets/second")
        print()
        print("  ╔═══════════════════════════════════════╗")
        print("  ║  YOUR HARDWARE IS WORKING!            ║")
        print("  ║                                       ║")
        print("  ║  Next step:                           ║")
        print("  ║  python step2_water_test.py            ║")
        print("  ╚═══════════════════════════════════════╝")
        return True
        
    elif csi_count > 0:
        print(f"  ⚠️ Got only {csi_count} CSI packets (expected 50+)")
        print("     Signal may be weak. Try:")
        print("     1. Move TX and RX closer together")
        print("     2. Make sure TX is powered on")
        print("     3. Check TX Serial Monitor shows 'TRANSMITTING'")
        return True
        
    else:
        print(f"  ❌ No CSI packets received!")
        print(f"     Got {other_count} other serial messages")
        print()
        print("  TROUBLESHOOTING:")
        print("    1. Is the TX ESP32 powered on?")
        print("       → Its Serial Monitor should show 'TRANSMITTING'")
        print("    2. Is the RX connected to TX's WiFi?")
        print("       → RX Serial Monitor should show 'Connected!'")
        print("    3. Are TX and RX close enough? (within 2 meters)")
        print("    4. Did you flash the correct firmware?")
        print("       → TX gets tx_firmware.ino")
        print("       → RX gets rx_firmware.ino")
        return False


if __name__ == '__main__':
    test_connection()
