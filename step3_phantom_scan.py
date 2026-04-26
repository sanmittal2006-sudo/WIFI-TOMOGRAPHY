#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║  STEP 3: FULLY AUTOMATED Phantom Scan                ║
║  Controls motor + collects CSI — no Enter pressing!  ║
╚══════════════════════════════════════════════════════╝

Usage:
    python step3_phantom_scan.py

What it does:
    1. Connects to RX ESP32 (CSI data)
    2. Connects to Arduino Uno (motor control)
    3. For each of 16 positions:
       - Tells Arduino to rotate 22.5°
       - Waits for motor to finish
       - Collects 100 CSI frames
    4. Saves .npz + .csv + .json
"""

import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import time
import os
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════
#   CONFIGURATION — CHANGE THESE TO YOUR PORTS
# ═══════════════════════════════════════════════════════
RX_PORT          = 'COM7'      # RX ESP32 port (CSI data)
MOTOR_PORT       = 'COM11'     # Arduino Uno port (motor)
BAUD_RATE        = 115200      # ESP-IDF default
MOTOR_BAUD       = 9600        # Arduino default
FRAMES_PER_POS   = 100         # Frames to collect per position
NUM_POSITIONS    = 16          # 16 × 22.5° = 360°
ANGLE_STEP       = 22.5        # Degrees per step
SETTLE_TIME      = 1.0         # Seconds to wait after motor stops
SAVE_DIR         = 'real_scans'
# ═══════════════════════════════════════════════════════

os.makedirs(SAVE_DIR, exist_ok=True)


def parse_csi(line):
    """Parse CSI_DATA line → complex array."""
    try:
        if not line.startswith('CSI_DATA'):
            return None, None, None
        parts = line.split(',', 5)
        rssi  = int(parts[2])
        noise = int(parts[3])
        bracket_start = line.find('[')
        bracket_end   = line.find(']')
        if bracket_start < 0 or bracket_end < 0:
            return None, None, None
        vals = [int(x) for x in line[bracket_start+1:bracket_end].split(',') if x.strip()]
        if len(vals) < 4:
            return None, None, None
        n_pairs = len(vals) // 2
        csi = np.array([vals[2*i] + 1j*vals[2*i+1] for i in range(n_pairs)])
        return csi, rssi, noise
    except:
        return None, None, None


def collect_position(ser, n_frames=100, timeout=20):
    """Collect N CSI frames at current position."""
    frames = []
    rssi_list = []
    noise_list = []
    ser.reset_input_buffer()
    time.sleep(0.3)
    t0 = time.time()

    while len(frames) < n_frames:
        if time.time() - t0 > timeout:
            print(f"      ⏰ Timeout — got {len(frames)}/{n_frames}")
            break
        try:
            raw  = ser.readline()
            line = raw.decode('utf-8', errors='ignore').strip()
            csi, rssi, noise = parse_csi(line)
            if csi is not None:
                frames.append(csi)
                rssi_list.append(rssi)
                noise_list.append(noise)
                if len(frames) % 25 == 0:
                    print(f"      [{len(frames)}/{n_frames}]", flush=True)
        except:
            continue

    if not frames:
        return None

    min_len = min(len(f) for f in frames)
    frames = np.array([f[:min_len] for f in frames])
    mean_csi = np.mean(frames, axis=0)

    return {
        'mean_csi':   mean_csi,
        'amplitude':  np.abs(mean_csi),
        'phase':      np.angle(mean_csi),
        'rssi_mean':  float(np.mean(rssi_list)),
        'rssi_std':   float(np.std(rssi_list)),
        'noise_mean': float(np.mean(noise_list)),
        'n_frames':   len(frames),
        'n_sub':      min_len,
        'timestamp':  datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    }


def motor_command(motor_ser, cmd, timeout=10):
    """Send command to Arduino, wait for response."""
    motor_ser.write(f"{cmd}\n".encode())
    t0 = time.time()
    while time.time() - t0 < timeout:
        if motor_ser.in_waiting > 0:
            response = motor_ser.readline().decode('utf-8', errors='ignore').strip()
            if response:
                return response
        time.sleep(0.05)
    return None


def run_full_scan(condition):
    """Fully automated 16-position scan."""
    
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  AUTOMATED SCAN: {condition.upper():34s}  ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Positions:    {NUM_POSITIONS} × {ANGLE_STEP}° = 360°               ║")
    print(f"║  Frames/pos:   {FRAMES_PER_POS}                                ║")
    print(f"║  FULLY AUTOMATIC — no Enter pressing!          ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    
    # ── Connect to RX ESP32 ────────────────────────────────
    print("── Connecting to RX ESP32 (CSI) ──")
    try:
        rx = serial.Serial(RX_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"  ✅ RX connected on {RX_PORT}")
    except Exception as e:
        print(f"  ❌ RX connect failed: {e}")
        return None
    
    # Quick CSI test
    rx.reset_input_buffer()
    test_count = 0
    for _ in range(30):
        line = rx.readline().decode('utf-8', errors='ignore')
        if 'CSI_DATA' in line:
            test_count += 1
    if test_count == 0:
        print("  ❌ No CSI data! Check TX is powered on.")
        rx.close()
        return None
    print(f"  ✅ CSI active ({test_count} packets in test)")
    print()
    
    # ── Connect to Arduino (motor) ─────────────────────────
    print("── Connecting to Arduino (Motor) ──")
    try:
        motor = serial.Serial(MOTOR_PORT, MOTOR_BAUD, timeout=2)
        time.sleep(2)  # Arduino resets on serial connect
        
        # Wait for MOTOR_READY
        for _ in range(10):
            if motor.in_waiting > 0:
                resp = motor.readline().decode('utf-8', errors='ignore').strip()
                if 'MOTOR_READY' in resp or 'READY' in resp:
                    break
            time.sleep(0.5)
        
        # Test connection
        resp = motor_command(motor, "PING")
        if resp and 'PONG' in resp:
            print(f"  ✅ Motor connected on {MOTOR_PORT}")
        else:
            print(f"  ⚠️ Motor connected but no PONG (got: {resp})")
            print(f"     Continuing anyway...")
    except Exception as e:
        print(f"  ❌ Motor connect failed: {e}")
        print(f"     Check COM port. Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"       {p.device} — {p.description}")
        rx.close()
        return None
    print()
    
    # ── Phantom condition info ─────────────────────────────
    if condition == 'healthy':
        print("  CONDITION: HEALTHY — lungs empty (air only)")
    elif condition == 'mild':
        print("  CONDITION: MILD — 15ml water in right lung")
    elif condition == 'moderate':
        print("  CONDITION: MODERATE — 50ml water in right lung")
    elif condition == 'severe':
        print("  CONDITION: SEVERE — 150ml water in right lung")
    print()
    input("  ▶ Press ENTER once to start the fully automated scan...")
    print()
    
    # ── AUTOMATED SCAN ─────────────────────────────────────
    scan_start = datetime.now()
    all_data = {}
    csv_rows = []
    
    for pos in range(NUM_POSITIONS):
        angle = pos * ANGLE_STEP
        
        print(f"  ┌─────────────────────────────────────┐")
        print(f"  │  Position {pos+1:2d}/{NUM_POSITIONS}  |  Angle: {angle:6.1f}°    │")
        print(f"  └─────────────────────────────────────┘")
        
        # Move motor (skip for position 0 — already at start)
        if pos > 0:
            print(f"      🔄 Rotating 22.5°...", end="", flush=True)
            resp = motor_command(motor, "MOVE")
            if resp:
                print(f" ✅ ({resp})")
            else:
                print(f" ⚠️ No response (motor may still have moved)")
            
            # Let phantom settle after rotation
            print(f"      ⏳ Settling {SETTLE_TIME}s...", flush=True)
            time.sleep(SETTLE_TIME)
        
        # Collect CSI
        print(f"      📡 Collecting CSI...", flush=True)
        data = collect_position(rx, FRAMES_PER_POS)
        
        if data is None:
            print("      ⚠️ Failed! Retrying...")
            time.sleep(2)
            data = collect_position(rx, FRAMES_PER_POS)
        
        if data is not None:
            all_data[pos] = data
            print(f"      ✅ {data['n_frames']} frames | {data['n_sub']} sub | "
                  f"RSSI: {data['rssi_mean']:.1f} dBm | "
                  f"Amp: {np.mean(data['amplitude']):.2f}")
            
            # CSV row
            row = {
                'timestamp':     data['timestamp'],
                'condition':     condition,
                'position':      pos,
                'angle_deg':     angle,
                'rssi_dbm':      round(data['rssi_mean'], 2),
                'rssi_std':      round(data['rssi_std'], 2),
                'noise_dbm':     round(data['noise_mean'], 2),
                'n_frames':      data['n_frames'],
                'n_subcarriers': data['n_sub'],
                'amp_mean':      round(float(np.mean(data['amplitude'])), 4),
                'amp_max':       round(float(np.max(data['amplitude'])), 4),
            }
            for k in range(data['n_sub']):
                row[f'csi_real_{k:02d}'] = round(data['mean_csi'][k].real, 4)
                row[f'csi_imag_{k:02d}'] = round(data['mean_csi'][k].imag, 4)
            csv_rows.append(row)
        else:
            print("      ❌ FAILED — skipping")
        print()
    
    # Close connections
    rx.close()
    
    # Return motor to home
    print("  🏠 Returning motor to home position...")
    resp = motor_command(motor, "HOME", timeout=30)
    print(f"     {resp}")
    motor.close()
    
    # ── Save data ──────────────────────────────────────────
    ts_str = scan_start.strftime('%Y%m%d_%H%M%S')
    base_path = os.path.join(SAVE_DIR, f'{condition}_{ts_str}')
    
    # NPZ
    npz_data = {}
    for pos, d in all_data.items():
        npz_data[f'pos{pos:02d}_csi_real'] = d['mean_csi'].real
        npz_data[f'pos{pos:02d}_csi_imag'] = d['mean_csi'].imag
        npz_data[f'pos{pos:02d}_amplitude'] = d['amplitude']
        npz_data[f'pos{pos:02d}_phase'] = d['phase']
        npz_data[f'pos{pos:02d}_rssi'] = np.array([d['rssi_mean']])
    np.savez(base_path + '.npz', **npz_data)
    
    # CSV
    df = pd.DataFrame(csv_rows)
    df.to_csv(base_path + '.csv', index=False)
    df.to_csv(os.path.join(SAVE_DIR, f'{condition}.csv'), index=False)
    
    # JSON summary
    summary = {
        'scan_info': {
            'condition':        condition,
            'start_time':       scan_start.strftime('%Y-%m-%d %H:%M:%S'),
            'end_time':         datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': round((datetime.now() - scan_start).total_seconds()),
            'positions_done':   len(all_data),
            'positions_total':  NUM_POSITIONS,
            'frames_per_pos':   FRAMES_PER_POS,
            'automated':        True,
        },
        'hardware': {
            'rx_port':          RX_PORT,
            'motor_port':       MOTOR_PORT,
            'phantom_diameter': '20 cm',
            'fill_height':      '13 cm',
            'antenna_distance': '15 cm from shaft',
        },
        'quality_metrics': {
            'positions_successful': len(all_data),
            'positions_failed':     NUM_POSITIONS - len(all_data),
            'mean_rssi_dbm':        round(np.mean([d['rssi_mean'] for d in all_data.values()]), 1) if all_data else 0,
        },
        'per_position': {}
    }
    for pos, d in all_data.items():
        summary['per_position'][f'pos_{pos:02d}_{pos*ANGLE_STEP:.1f}deg'] = {
            'timestamp':    d['timestamp'],
            'rssi_dbm':     round(d['rssi_mean'], 1),
            'n_frames':     d['n_frames'],
            'amp_mean':     round(float(np.mean(d['amplitude'])), 3),
        }
    with open(base_path + '_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # ── Final summary ──────────────────────────────────────
    duration = round((datetime.now() - scan_start).total_seconds())
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  ✅ SCAN COMPLETE: {condition.upper():32s}   ║")
    print(f"║  Positions: {len(all_data):2d}/{NUM_POSITIONS}  |  Duration: {duration}s           ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  {base_path}.npz")
    print(f"║  {base_path}.csv")
    print(f"║  {base_path}_summary.json")
    print("╚══════════════════════════════════════════════════════╝")
    
    return base_path


if __name__ == '__main__':
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Wi-Fi Tomography — AUTOMATED Phantom Scanner       ║")
    print("║  Motor + CSI — fully automatic!                     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    
    ports = serial.tools.list_ports.comports()
    print("  COM Ports:")
    for p in ports:
        print(f"    {p.device:8s} — {p.description}")
    print()
    
    print("  Which condition?")
    print("    1. healthy   — lungs empty")
    print("    2. mild      — 15ml water in right lung")
    print("    3. moderate  — 50ml water in right lung")
    print("    4. severe    — 150ml water in right lung")
    print()
    
    choice = input("  Enter 1-4: ").strip()
    conditions = {'1':'healthy', '2':'mild', '3':'moderate', '4':'severe'}
    condition = conditions.get(choice, 'healthy')
    
    run_full_scan(condition)
