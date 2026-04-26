#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║  STEP 2: Water Bottle Test                           ║
║  Verify that CSI changes when object is placed       ║
║  between TX and RX                                   ║
╚══════════════════════════════════════════════════════╝

Usage:
    python step2_water_test.py

What it does:
    1. Records CSI with NO object (air baseline)
    2. Asks you to place a water bottle between TX and RX
    3. Records CSI WITH the water bottle
    4. Shows the DIFFERENCE — proves the system detects objects
    5. Saves a comparison plot
"""

import serial
import numpy as np
import matplotlib.pyplot as plt
import time
import os

# ═══════════════════════════════════════════════════════
#   CHANGE THIS TO YOUR COM PORT
# ═══════════════════════════════════════════════════════
RX_PORT   = 'COM7'
BAUD_RATE = 115200
FRAMES    = 200       # Frames to average (more = smoother)
# ═══════════════════════════════════════════════════════


def parse_csi(line):
    """Parse one CSI_DATA line → complex array."""
    try:
        if not line.startswith('CSI_DATA'):
            return None, None
        
        parts = line.split(',', 5)
        rssi  = int(parts[2])
        
        bracket_start = line.find('[')
        bracket_end   = line.find(']')
        if bracket_start < 0 or bracket_end < 0:
            return None, None
        
        vals = [int(x) for x in line[bracket_start+1:bracket_end].split(',') if x.strip()]
        
        if len(vals) < 56:  # Need at least 28 subcarrier pairs
            return None, None
        
        # Pair up: real, imag, real, imag, ...
        n_pairs = len(vals) // 2
        csi = np.array([vals[2*i] + 1j*vals[2*i+1] for i in range(n_pairs)])
        
        return csi, rssi
    except:
        return None, None


def collect_frames(ser, n_frames, label=""):
    """Collect N CSI frames and return averaged amplitude + phase."""
    frames = []
    rssi_list = []
    
    ser.reset_input_buffer()
    time.sleep(0.2)
    
    print(f"    Collecting {n_frames} frames...", end="", flush=True)
    t0 = time.time()
    
    while len(frames) < n_frames:
        if time.time() - t0 > 30:
            print(f"\n    ⏰ Timeout! Got {len(frames)}/{n_frames}")
            break
        try:
            raw  = ser.readline()
            line = raw.decode('utf-8', errors='ignore').strip()
            csi, rssi = parse_csi(line)
            if csi is not None:
                frames.append(csi)
                rssi_list.append(rssi)
                if len(frames) % 50 == 0:
                    print(f" {len(frames)}", end="", flush=True)
        except:
            continue
    
    if not frames:
        print(" ❌ No data!")
        return None
    
    # Make all frames same length
    min_len = min(len(f) for f in frames)
    frames = np.array([f[:min_len] for f in frames])
    
    mean_csi = np.mean(frames, axis=0)
    
    print(f" ✅ ({len(frames)} frames, {min_len} subcarriers)")
    print(f"    Mean RSSI: {np.mean(rssi_list):.1f} dBm")
    
    return {
        'csi':       mean_csi,
        'amplitude': np.abs(mean_csi),
        'phase':     np.angle(mean_csi),
        'rssi':      np.mean(rssi_list),
        'n_sub':     min_len
    }


def main():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Wi-Fi Tomography — Water Bottle Test                ║")
    print("║  Does the system detect an object?                   ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Connect
    print("── Connecting to RX ──")
    try:
        ser = serial.Serial(RX_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"  ✅ Connected to {RX_PORT}")
    except Exception as e:
        print(f"  ❌ Cannot connect: {e}")
        print("  Run step1_test_connection.py first!")
        return
    print()

    # ── Measurement 1: AIR (no object) ────────────────────
    print("── MEASUREMENT 1: Baseline (AIR) ──")
    print("  Make sure NOTHING is between TX and RX")
    input("  ▶ Press ENTER when ready...")
    print()
    air = collect_frames(ser, FRAMES, "AIR")
    if air is None:
        ser.close()
        return
    print()

    # ── Measurement 2: WATER BOTTLE ───────────────────────
    print("── MEASUREMENT 2: With Object ──")
    print("  Place a WATER BOTTLE (or your phantom) between TX and RX")
    input("  ▶ Press ENTER when object is in place...")
    print()
    obj = collect_frames(ser, FRAMES, "OBJECT")
    if obj is None:
        ser.close()
        return
    
    ser.close()
    print()

    # ── Compare ────────────────────────────────────────────
    min_n = min(air['n_sub'], obj['n_sub'])
    air_amp = air['amplitude'][:min_n]
    obj_amp = obj['amplitude'][:min_n]
    diff    = obj_amp - air_amp
    
    # Statistics
    mean_change = np.mean(np.abs(diff))
    max_change  = np.max(np.abs(diff))
    pct_change  = 100 * mean_change / (np.mean(air_amp) + 1e-10)

    print("── RESULTS ──")
    print()
    print(f"  Air baseline amplitude:  mean={np.mean(air_amp):.2f}")
    print(f"  Object amplitude:        mean={np.mean(obj_amp):.2f}")
    print(f"  Difference:              mean={mean_change:.2f}  max={max_change:.2f}")
    print(f"  Change:                  {pct_change:.1f}%")
    print()
    
    if pct_change > 5:
        print("  ╔═══════════════════════════════════════╗")
        print("  ║  ✅ OBJECT DETECTED!                  ║")
        print(f"  ║  CSI changed by {pct_change:.1f}%               ║")
        print("  ║  Your system can detect objects!      ║")
        print("  ║                                       ║")
        print("  ║  Next step:                           ║")
        print("  ║  python step3_phantom_scan.py          ║")
        print("  ╚═══════════════════════════════════════╝")
    elif pct_change > 1:
        print("  ⚠️ Small change detected ({pct_change:.1f}%)")
        print("     Try a larger object or move TX/RX closer")
    else:
        print("  ❌ No significant change detected")
        print("     → Make sure the object is between TX and RX")
        print("     → Use a bigger water bottle")
        print("     → Move TX and RX closer together")

    # ── Plot ───────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    subcarrier_idx = np.arange(min_n)
    
    # Amplitude comparison
    axes[0].plot(subcarrier_idx, air_amp, 'b-', linewidth=1.5, label='Air (no object)')
    axes[0].plot(subcarrier_idx, obj_amp, 'r-', linewidth=1.5, label='With object')
    axes[0].set_xlabel('Subcarrier Index')
    axes[0].set_ylabel('CSI Amplitude')
    axes[0].set_title('CSI Amplitude Comparison', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Difference
    colors = ['green' if d > 0 else 'red' for d in diff]
    axes[1].bar(subcarrier_idx, diff, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Subcarrier Index')
    axes[1].set_ylabel('Amplitude Change')
    axes[1].set_title(f'CSI Change Due to Object (mean={mean_change:.2f}, {pct_change:.1f}%)',
                      fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    # Phase comparison
    axes[2].plot(subcarrier_idx, air['phase'][:min_n], 'b-', linewidth=1.5, label='Air')
    axes[2].plot(subcarrier_idx, obj['phase'][:min_n], 'r-', linewidth=1.5, label='With object')
    axes[2].set_xlabel('Subcarrier Index')
    axes[2].set_ylabel('Phase (radians)')
    axes[2].set_title('CSI Phase Comparison', fontsize=13, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('test_results', exist_ok=True)
    plt.savefig('test_results/water_bottle_test.png', dpi=150, bbox_inches='tight')
    print(f"\n  📊 Plot saved: test_results/water_bottle_test.png")
    plt.show()


if __name__ == '__main__':
    main()
