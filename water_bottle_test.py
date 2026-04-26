"""
Test: Does a water bottle change the CSI?
Measures CSI with and without an object, shows the DIFFERENCE.
"""
import serial
import numpy as np
import matplotlib.pyplot as plt
import time

PORT = 'COM7'
BAUD = 921600
NUM_FRAMES = 200  # Average 200 frames for less noise

def parse_csi(line):
    """Parse CSI from one serial line"""
    try:
        if 'CSI_DATA' not in line:
            return None
        bracket_start = line.index('[')
        bracket_end = line.index(']')
        data_str = line[bracket_start+1:bracket_end]
        numbers = [int(x.strip()) for x in data_str.split(',') if x.strip()]
        if len(numbers) < 10:
            return None
        csi = np.array([numbers[i] + 1j * numbers[i+1] for i in range(0, len(numbers)-1, 2)])
        return csi
    except:
        return None

def collect_average(ser, num_frames=200, label=""):
    """Collect and average multiple CSI frames"""
    frames = []
    ser.reset_input_buffer()
    time.sleep(0.3)
    
    print(f"  Collecting {num_frames} frames for '{label}'...")
    start = time.time()
    
    while len(frames) < num_frames and time.time() - start < 30:
        try:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            csi = parse_csi(line)
            if csi is not None and len(csi) > 20:
                frames.append(csi)
        except:
            continue
    
    if len(frames) < 10:
        print(f"  ⚠️ Only got {len(frames)} frames!")
        return None
    
    # Make all same length
    min_len = min(len(f) for f in frames)
    frames = np.array([f[:min_len] for f in frames])
    
    mean_csi = np.mean(frames, axis=0)
    print(f"  ✅ Got {len(frames)} frames, {min_len} subcarriers")
    return mean_csi

# ============ MAIN ============
print("=" * 60)
print("  WATER BOTTLE TEST")
print("  Does CSI change when object is placed between ESP32s?")
print("=" * 60)

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)
print("Connected!\n")

# MEASUREMENT 1: Nothing between ESP32s
print("📡 MEASUREMENT 1: EMPTY (nothing between ESP32s)")
input("   Remove everything between ESP32s. Press ENTER...")
empty = collect_average(ser, NUM_FRAMES, "empty")

# MEASUREMENT 2: Water bottle between ESP32s
print("\n📡 MEASUREMENT 2: WATER BOTTLE between ESP32s")
input("   Place water bottle DIRECTLY between the antennas. Press ENTER...")
bottle = collect_average(ser, NUM_FRAMES, "water bottle")

# MEASUREMENT 3: Hand between ESP32s
print("\n📡 MEASUREMENT 3: YOUR HAND between ESP32s")
input("   Hold your hand flat between the antennas. Press ENTER...")
hand = collect_average(ser, NUM_FRAMES, "hand")

ser.close()

# ============ COMPARE ============
if empty is not None and bottle is not None and hand is not None:
    # Make sure all same length
    min_len = min(len(empty), len(bottle), len(hand))
    empty = empty[:min_len]
    bottle = bottle[:min_len]
    hand = hand[:min_len]
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Plot 1: Raw amplitudes
    axes[0].plot(np.abs(empty), 'g-', label='Empty (no object)', linewidth=2)
    axes[0].plot(np.abs(bottle), 'b-', label='Water bottle', linewidth=2)
    axes[0].plot(np.abs(hand), 'r-', label='Hand', linewidth=2)
    axes[0].set_xlabel('Subcarrier Index')
    axes[0].set_ylabel('CSI Amplitude')
    axes[0].set_title('RAW CSI Amplitude — Looks Similar (This Is Expected!)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: DIFFERENCE in amplitude (THIS is what matters!)
    diff_bottle = np.abs(bottle) - np.abs(empty)
    diff_hand = np.abs(hand) - np.abs(empty)
    axes[1].plot(diff_bottle, 'b-', label='Bottle - Empty', linewidth=2)
    axes[1].plot(diff_hand, 'r-', label='Hand - Empty', linewidth=2)
    axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Subcarrier Index')
    axes[1].set_ylabel('Amplitude Difference')
    axes[1].set_title('DIFFERENCE in Amplitude — THIS Shows Object Detection!')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: DIFFERENCE in phase
    phase_diff_bottle = np.angle(bottle) - np.angle(empty)
    phase_diff_hand = np.angle(hand) - np.angle(empty)
    # Wrap to [-pi, pi]
    phase_diff_bottle = np.angle(np.exp(1j * phase_diff_bottle))
    phase_diff_hand = np.angle(np.exp(1j * phase_diff_hand))
    axes[2].plot(phase_diff_bottle, 'b-', label='Bottle - Empty', linewidth=2)
    axes[2].plot(phase_diff_hand, 'r-', label='Hand - Empty', linewidth=2)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[2].set_xlabel('Subcarrier Index')
    axes[2].set_ylabel('Phase Difference (radians)')
    axes[2].set_title('DIFFERENCE in Phase — THIS Shows Object Detection!')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('water_bottle_test.png', dpi=150)
    plt.show()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  Mean amplitude change (bottle): {np.mean(np.abs(diff_bottle)):.3f}")
    print(f"  Mean amplitude change (hand):   {np.mean(np.abs(diff_hand)):.3f}")
    print(f"  Mean phase change (bottle):     {np.mean(np.abs(phase_diff_bottle)):.3f} rad")
    print(f"  Mean phase change (hand):       {np.mean(np.abs(phase_diff_hand)):.3f} rad")
    
    if np.mean(np.abs(diff_hand)) > np.mean(np.abs(diff_bottle)):
        print("\n  ✅ Hand causes MORE change than bottle (expected — hand is bigger!)")
    print("\n  📊 Saved: water_bottle_test.png")
else:
    print("❌ Failed to collect all measurements. Check ESP32 connection.")
