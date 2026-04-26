"""
CSI Data Collection Script for Wi-Fi Tomography
Reads CSI from ESP32 RX, saves 16 positions at 22.5° each
"""
import serial
import numpy as np
import time
import os
import sys

# ====== CHANGE THESE TO MATCH YOUR SETUP ======
RX_PORT = 'COM7'        # Your ESP32 RX port
BAUD_RATE = 115200       # ESP-IDF default baud rate
FRAMES_PER_POS = 100     # Collect 100 frames per position (average for noise reduction)
NUM_POSITIONS = 16       # 16 × 22.5° = 360°
SAVE_FOLDER = 'csi_data'
# ================================================

def parse_csi_line(line):
    """
    Parse one CSI_DATA line from ESP32 serial output.
    Extracts the complex CSI values (real + imaginary pairs).
    """
    try:
        if 'CSI_DATA' not in line:
            return None, None
        
        # Extract RSSI (4th field after splitting by comma)
        parts = line.split(',')
        rssi = int(parts[3])  # RSSI value
        
        # Extract CSI data from square brackets
        bracket_start = line.index('[')
        bracket_end = line.index(']')
        data_str = line[bracket_start+1:bracket_end]
        
        # Parse all numbers
        numbers = [int(x.strip()) for x in data_str.split(',') if x.strip()]
        
        if len(numbers) < 2:
            return None, None
        
        # Convert to complex: pairs of (real, imaginary)
        num_subcarriers = len(numbers) // 2
        csi = np.array([
            numbers[i] + 1j * numbers[i+1] 
            for i in range(0, len(numbers) - 1, 2)
        ])
        
        return csi, rssi
        
    except Exception as e:
        return None, None


def collect_one_position(ser, num_frames=100, timeout=10):
    """Collect multiple CSI frames at the current phantom position."""
    frames = []
    rssi_values = []
    start = time.time()
    
    # Clear old data from buffer
    ser.reset_input_buffer()
    time.sleep(0.2)
    
    while len(frames) < num_frames:
        if time.time() - start > timeout:
            print(f"    ⏰ Timeout after {timeout}s. Got {len(frames)}/{num_frames} frames.")
            break
        
        try:
            raw = ser.readline()
            line = raw.decode('utf-8', errors='ignore').strip()
            
            csi, rssi = parse_csi_line(line)
            
            if csi is not None and len(csi) >= 28:  # At least 28 subcarriers
                frames.append(csi)
                rssi_values.append(rssi)
        except Exception:
            continue
    
    if len(frames) == 0:
        return None
    
    # Make all frames same length (use minimum length)
    min_len = min(len(f) for f in frames)
    frames = [f[:min_len] for f in frames]
    frames = np.array(frames)  # Shape: [num_frames, num_subcarriers]
    
    # Average across frames to reduce noise
    mean_csi = np.mean(frames, axis=0)
    
    return {
        'raw_frames': frames,
        'mean_csi': mean_csi,
        'amplitude': np.abs(mean_csi),
        'phase': np.angle(mean_csi),
        'rssi_mean': np.mean(rssi_values),
        'num_frames': len(frames),
        'num_subcarriers': min_len
    }


def run_scan(phantom_name='healthy'):
    """Run a complete 16-position scan."""
    
    os.makedirs(SAVE_FOLDER, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  Wi-Fi Tomography — CSI Data Collection")
    print(f"  Phantom: {phantom_name}")
    print(f"  Positions: {NUM_POSITIONS} × 22.5° = 360°")
    print(f"  Frames per position: {FRAMES_PER_POS}")
    print(f"{'='*60}")
    
    # Connect to ESP32 RX
    print(f"\n📡 Connecting to ESP32-RX on {RX_PORT}...")
    try:
        ser = serial.Serial(RX_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for connection to stabilize
        print(f"✅ Connected!")
    except Exception as e:
        print(f"❌ ERROR: Cannot connect to {RX_PORT}")
        print(f"   Make sure ESP32-RX is plugged in and no other program uses {RX_PORT}")
        print(f"   Close any ESP-IDF monitor windows first!")
        print(f"   Error: {e}")
        return None
    
    # Test: read a few lines to make sure data is coming
    print(f"\n🔍 Testing CSI reception...")
    test_count = 0
    for _ in range(20):
        line = ser.readline().decode('utf-8', errors='ignore')
        if 'CSI_DATA' in line:
            test_count += 1
    
    if test_count == 0:
        print(f"❌ No CSI data received! Make sure:")
        print(f"   1. ESP32-TX is plugged in and powered")
        print(f"   2. ESP32-RX is running (check LED)")
        print(f"   3. Both ESP32s are close enough (within 1 meter)")
        ser.close()
        return None
    
    print(f"✅ Receiving CSI data ({test_count} frames in test)")
    
    # Start collecting
    all_data = {}
    
    print(f"\n{'='*60}")
    print(f"  STARTING SCAN — Place phantom in center")
    print(f"  Mark position 0° on the turntable")
    print(f"{'='*60}")
    
    input("\n▶ Press ENTER when phantom is in position and ready to start...")
    
    for pos in range(NUM_POSITIONS):
        angle = pos * 22.5
        
        print(f"\n📍 POSITION {pos+1}/{NUM_POSITIONS} — Angle: {angle}°")
        print(f"   Collecting {FRAMES_PER_POS} CSI frames...")
        
        data = collect_one_position(ser, FRAMES_PER_POS)
        
        if data is None:
            print(f"   ⚠️ FAILED! Retrying in 2 seconds...")
            time.sleep(2)
            data = collect_one_position(ser, FRAMES_PER_POS)
        
        if data is not None:
            all_data[f'pos_{pos:02d}'] = data
            print(f"   ✅ Got {data['num_frames']} frames")
            print(f"      Subcarriers: {data['num_subcarriers']}")
            print(f"      Mean amplitude: {np.mean(data['amplitude']):.2f}")
            print(f"      Mean RSSI: {data['rssi_mean']:.1f} dBm")
        else:
            print(f"   ❌ FAILED AGAIN — skipping this position")
        
        # Ask to rotate (except at last position)
        if pos < NUM_POSITIONS - 1:
            next_angle = (pos + 1) * 22.5
            print(f"\n   🔄 ROTATE phantom to {next_angle}° (turn 22.5° clockwise)")
            input(f"   ▶ Press ENTER after rotating...")
    
    # Save data
    save_dict = {}
    for key, val in all_data.items():
        save_dict[f'{key}_mean_csi_real'] = val['mean_csi'].real
        save_dict[f'{key}_mean_csi_imag'] = val['mean_csi'].imag
        save_dict[f'{key}_amplitude'] = val['amplitude']
        save_dict[f'{key}_phase'] = val['phase']
        save_dict[f'{key}_rssi'] = np.array([val['rssi_mean']])
    
    filepath = os.path.join(SAVE_FOLDER, f'{phantom_name}.npz')
    np.savez(filepath, **save_dict)
    
    print(f"\n{'='*60}")
    print(f"  ✅ SCAN COMPLETE!")
    print(f"  📊 Collected: {len(all_data)}/{NUM_POSITIONS} positions")
    print(f"  💾 Saved to: {filepath}")
    print(f"{'='*60}")
    
    ser.close()
    return all_data


def quick_plot(phantom_name='healthy'):
    """Quick plot of collected data to verify it looks right."""
    import matplotlib.pyplot as plt
    
    filepath = os.path.join(SAVE_FOLDER, f'{phantom_name}.npz')
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
    
    data = np.load(filepath)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot amplitude for all positions
    for i in range(NUM_POSITIONS):
        key = f'pos_{i:02d}_amplitude'
        if key in data:
            axes[0].plot(data[key], alpha=0.5, label=f'{i*22.5}°')
    
    axes[0].set_xlabel('Subcarrier Index')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(f'CSI Amplitude — {phantom_name}')
    axes[0].legend(fontsize=7, ncol=4)
    axes[0].grid(True, alpha=0.3)
    
    # Plot phase for all positions
    for i in range(NUM_POSITIONS):
        key = f'pos_{i:02d}_phase'
        if key in data:
            axes[1].plot(data[key], alpha=0.5, label=f'{i*22.5}°')
    
    axes[1].set_xlabel('Subcarrier Index')
    axes[1].set_ylabel('Phase (radians)')
    axes[1].set_title(f'CSI Phase — {phantom_name}')
    axes[1].legend(fontsize=7, ncol=4)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plotfile = os.path.join(SAVE_FOLDER, f'{phantom_name}_plot.png')
    plt.savefig(plotfile, dpi=150)
    print(f"📊 Plot saved: {plotfile}")
    plt.show()


# ============ MAIN ============
if __name__ == '__main__':
    print("\n🔬 Wi-Fi Microwave Tomography — Data Collection\n")
    print("Available commands:")
    print("  1. Collect data (new scan)")
    print("  2. Plot existing data")
    print("  3. Exit")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        print("\nPhantom options: healthy, mild, moderate, severe")
        name = input("Enter phantom name: ").strip().lower()
        if name not in ['healthy', 'mild', 'moderate', 'severe']:
            print(f"Using custom name: {name}")
        run_scan(phantom_name=name)
        
        # Auto-plot after collection
        plot_choice = input("\nPlot the data now? (y/n): ").strip().lower()
        if plot_choice == 'y':
            quick_plot(name)
    
    elif choice == '2':
        name = input("Enter phantom name to plot: ").strip().lower()
        quick_plot(name)
    
    else:
        print("Bye!")
