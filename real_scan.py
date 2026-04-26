#!/usr/bin/env python3
"""
Wi-Fi Tomography — Real Phantom Scanner
========================================
Use this after flashing ESP32 firmware.
Scans the physical phantom at 16 positions,
saves data in both human-readable and machine-readable formats.

Usage:
    python real_scan.py

Author: Wi-Fi Tomography Project
Hardware: 2x ESP32-S3 DevKitC-1 N16R8
"""

import serial
import serial.tools.list_ports
import numpy as np
import pandas as pd
import time
import os
import json
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════
#   CHANGE THESE TO YOUR SETUP
# ═══════════════════════════════════════════════════════════════════════
RX_PORT       = 'COM7'      # RX ESP32 port (check Device Manager)
BAUD_RATE     = 921600      # Must match rx_firmware
FRAMES        = 100         # Frames per position (more = less noise)
N_POS         = 16          # Number of scan positions
SAVE_DIR      = 'real_scans'
# ═══════════════════════════════════════════════════════════════════════

os.makedirs(SAVE_DIR, exist_ok=True)

# ─── Helper: list available COM ports ──────────────────────────────────
def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("  ❌ No COM ports found!")
        return
    print("  Available COM ports:")
    for p in ports:
        print(f"    {p.device} — {p.description}")

# ─── Parse one CSI line from serial ────────────────────────────────────
def parse_csi_line(line):
    """
    Parse: CSI_DATA,seq,rssi,rate,noise,len,[r0,i0,r1,i1,...]
    Returns: (amplitude array, phase array, rssi) or None
    """
    try:
        if not line.startswith('CSI_DATA'):
            return None
        parts  = line.split(',', 6)
        rssi   = int(parts[2])
        noise  = int(parts[4])

        # Extract values from brackets
        bracket = parts[6].strip()
        bracket = bracket[1:bracket.index(']')]
        vals    = [int(x) for x in bracket.split(',') if x.strip()]

        if len(vals) < 4:
            return None

        # Pair up real/imag: [r0, i0, r1, i1, ...]
        n_pairs = len(vals) // 2
        real    = np.array(vals[0::2][:n_pairs], dtype=float)
        imag    = np.array(vals[1::2][:n_pairs], dtype=float)
        csi     = real + 1j * imag

        return {
            'amplitude': np.abs(csi),
            'phase':     np.angle(csi),
            'csi':       csi,
            'rssi':      rssi,
            'noise':     noise,
            'n_sub':     n_pairs
        }
    except Exception:
        return None

# ─── Collect N frames at current position ──────────────────────────────
def collect_position(ser, n_frames=100, timeout=15):
    frames   = []
    rss_vals = []
    ser.reset_input_buffer()
    t0 = time.time()

    while len(frames) < n_frames:
        if time.time() - t0 > timeout:
            print(f"      ⏰ Timeout — got {len(frames)}/{n_frames}")
            break
        try:
            raw  = ser.readline()
            line = raw.decode('utf-8', errors='ignore').strip()
            data = parse_csi_line(line)
            if data and data['n_sub'] >= 28:
                frames.append(data['csi'])
                rss_vals.append(data['rssi'])
        except Exception:
            continue

    if not frames:
        return None

    min_len = min(len(f) for f in frames)
    frames  = np.array([f[:min_len] for f in frames])   # [N, sub]
    mean    = np.mean(frames, axis=0)                    # average frames

    return {
        'mean_csi':  mean,
        'amplitude': np.abs(mean),
        'phase':     np.angle(mean),
        'rssi':      float(np.mean(rss_vals)),
        'n_frames':  len(frames),
        'n_sub':     min_len
    }

# ─── Run a full 16-position scan ───────────────────────────────────────
def run_scan(condition_name):
    print(f"\n{'═'*60}")
    print(f"  SCAN: {condition_name.upper()}")
    print(f"  Positions: {N_POS}  ×  22.5°  =  360°")
    print(f"  Frames per position: {FRAMES}")
    print(f"{'═'*60}")

    # Connect
    print(f"\n  Connecting to ESP32-RX on {RX_PORT}...")
    try:
        ser = serial.Serial(RX_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print("  ✅ Connected!")
    except Exception as e:
        print(f"  ❌ Cannot connect: {e}")
        print("  → Check COM port number in Device Manager")
        list_ports()
        return None

    # Test reception
    print("  Testing CSI reception...")
    count = 0
    for _ in range(30):
        line = ser.readline().decode('utf-8', errors='ignore')
        if 'CSI_DATA' in line:
            count += 1
    if count == 0:
        print("  ❌ No CSI received! Is TX ESP32 powered and nearby?")
        ser.close()
        return None
    print(f"  ✅ Receiving CSI ({count} packets in test)")

    # Scan all positions
    scan_data    = {}
    csv_rows     = []
    scan_start   = datetime.now()

    input(f"\n  ▶ Place phantom at 0°. Press ENTER to start scan...")

    for pos in range(N_POS):
        angle = pos * 22.5
        ts    = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

        print(f"\n  📍 Position {pos+1:2d}/{N_POS}  |  {angle:6.1f}°  |  {ts}")
        data = collect_position(ser, FRAMES)

        if data is None:
            print("     ⚠️ Failed — retrying...")
            data = collect_position(ser, FRAMES)

        if data:
            scan_data[pos] = data
            amp_mean = float(np.mean(data['amplitude']))
            amp_max  = float(np.max(data['amplitude']))

            # ── Human readable summary ──────────────────────────────
            print(f"     ✅ Frames:     {data['n_frames']}")
            print(f"        Sub-carriers: {data['n_sub']}")
            print(f"        RSSI:       {data['rssi']:.1f} dBm")
            print(f"        Amplitude:  mean={amp_mean:.2f}  max={amp_max:.2f}")
            print(f"        Phase range:{np.min(data['phase']):.2f} to {np.max(data['phase']):.2f} rad")

            # ── Machine readable CSV row ────────────────────────────
            row = {
                'timestamp':      ts,
                'condition':      condition_name,
                'position':       pos,
                'angle_deg':      angle,
                'rssi_dbm':       round(data['rssi'], 2),
                'n_frames':       data['n_frames'],
                'n_subcarriers':  data['n_sub'],
                'amp_mean':       round(amp_mean, 4),
                'amp_max':        round(amp_max, 4),
            }
            # Add all subcarrier values
            for k in range(data['n_sub']):
                row[f'csi_real_{k:02d}'] = round(data['mean_csi'][k].real, 4)
                row[f'csi_imag_{k:02d}'] = round(data['mean_csi'][k].imag, 4)
            csv_rows.append(row)
        else:
            print("     ❌ FAILED — skipping this position")

        # Prompt rotation
        if pos < N_POS - 1:
            next_angle = (pos + 1) * 22.5
            print(f"\n  🔄 Rotate phantom to {next_angle:.1f}° (22.5° clockwise)")
            input("  ▶ Press ENTER after rotating...")

    ser.close()

    # ── Save machine readable (.npz + .csv) ────────────────────────
    timestamp_str = scan_start.strftime('%Y%m%d_%H%M%S')
    base          = os.path.join(SAVE_DIR, f'{condition_name}_{timestamp_str}')

    # NPZ (for pipeline)
    npz_dict = {}
    for pos, d in scan_data.items():
        npz_dict[f'pos{pos:02d}_csi_real'] = d['mean_csi'].real
        npz_dict[f'pos{pos:02d}_csi_imag'] = d['mean_csi'].imag
        npz_dict[f'pos{pos:02d}_amp']       = d['amplitude']
        npz_dict[f'pos{pos:02d}_phase']     = d['phase']
        npz_dict[f'pos{pos:02d}_rssi']      = np.array([d['rssi']])
    np.savez(base + '.npz', **npz_dict)

    # CSV (human + machine readable)
    df = pd.DataFrame(csv_rows)
    df.to_csv(base + '.csv', index=False)

    # JSON summary (human readable overview)
    summary = {
        'scan_info': {
            'condition':       condition_name,
            'scan_time':       scan_start.strftime('%Y-%m-%d %H:%M:%S'),
            'positions_done':  len(scan_data),
            'positions_total': N_POS,
            'frames_per_pos':  FRAMES,
        },
        'hardware': {
            'rx_port':       RX_PORT,
            'baud_rate':     BAUD_RATE,
            'phantom_diam':  '20 cm',
            'fill_height':   '13 cm',
            'ant_distance':  '18 cm each side',
        },
        'per_position': {}
    }
    for pos, d in scan_data.items():
        summary['per_position'][f'pos_{pos:02d}_{pos*22.5:.1f}deg'] = {
            'rssi_dbm':    round(d['rssi'], 1),
            'n_frames':    d['n_frames'],
            'n_sub':       d['n_sub'],
            'amp_mean':    round(float(np.mean(d['amplitude'])), 3),
        }
    with open(base + '_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ── Print final summary ─────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  ✅ SCAN COMPLETE: {condition_name.upper()}")
    print(f"  Positions collected: {len(scan_data)}/{N_POS}")
    print(f"  Files saved:")
    print(f"    {base}.npz         ← for pipeline (BIM/U-Net)")
    print(f"    {base}.csv         ← spreadsheet / dashboard")
    print(f"    {base}_summary.json ← human readable overview")
    print(f"{'═'*60}")

    return base

# ─── MAIN ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n╔══════════════════════════════════════════╗")
    print("║  Wi-Fi Tomography — Real Phantom Scanner  ║")
    print("║  Physical phantom + ESP32-S3             ║")
    print("╚══════════════════════════════════════════╝\n")

    print("Available COM ports:")
    list_ports()

    print("\nSelect scan condition:")
    print("  1. healthy   (no water in lungs)")
    print("  2. mild      (15ml water in right lung)")
    print("  3. moderate  (50ml water in right lung)")
    print("  4. severe    (150ml water in right lung)")
    print("  5. custom name")

    choice = input("\nEnter choice (1-5): ").strip()
    names  = {'1':'healthy', '2':'mild', '3':'moderate', '4':'severe'}

    if choice in names:
        name = names[choice]
    elif choice == '5':
        name = input("Enter custom name: ").strip()
    else:
        name = 'scan'

    saved_base = run_scan(name)

    if saved_base:
        print(f"\n  Next step: run reconstruction pipeline:")
        print(f"  python reconstruct_real.py --scan {saved_base}.npz\n")
