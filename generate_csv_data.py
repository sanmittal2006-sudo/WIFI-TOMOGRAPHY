#!/usr/bin/env python3
"""
Wi-Fi Tomography — Realistic CSI Data Generator
================================================
Generates CSV files for all 4 phantom conditions:
  healthy, mild, moderate, severe

Each CSV contains:
  - timestamp
  - scan position (0-15, angle in degrees)
  - RSSI (dBm)
  - CSI real values for 52 subcarriers
  - CSI imaginary values for 52 subcarriers
  - setup metadata in header

Output: csi_data/
  healthy.csv, mild.csv, moderate.csv, severe.csv
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

# ── Setup ──────────────────────────────────────────────────────────────────
OUT = "csi_data"
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)

# Physical setup parameters
N_POS        = 16          # 16 rotation positions
N_SUBCARRIERS = 52         # ESP32 802.11n: 52 active subcarriers
FRAMES_PER_POS = 100       # 100 frames averaged per position
ANTENNA_DIST = 0.18        # 18cm TX-RX distance from phantom center
FREQ_GHz     = 2.4         # 2.4 GHz Wi-Fi
PHANTOM_DIAM = 0.20        # 20cm phantom diameter
DOMAIN       = 0.30        # 30cm domain
N            = 32          # 32x32 grid
DX           = DOMAIN / N
K0           = 2 * np.pi * FREQ_GHz * 1e9 / 3e8

# Dielectric properties at 2.4 GHz
EPS = {
    'chest_wall': 45.0,    # water+glycerine mixture
    'lung_air':    1.0,    # empty air cylinder
    'heart':      60.0,    # heart tissue
    'edema': {
        'none':     1.0,   # no fluid
        'mild':    50.0,   # mild fluid accumulation
        'moderate':65.0,   # moderate edema
        'severe':  78.0,   # severe (nearly pure water)
    }
}

# ── Phantom model ──────────────────────────────────────────────────────────
def make_phantom(condition):
    x = np.linspace(-DOMAIN/2+DX/2, DOMAIN/2-DX/2, N)
    xx, yy = np.meshgrid(x, x)
    eps = np.ones((N, N))

    # Chest wall
    eps[((xx/0.13)**2 + (yy/0.11)**2) <= 1.0] = EPS['chest_wall']
    # Left lung (air)
    eps[((xx+0.045)/0.05)**2 + (yy/0.07)**2 <= 1.0] = EPS['lung_air']
    # Right lung (air by default)
    eps[((xx-0.045)/0.05)**2 + (yy/0.07)**2 <= 1.0] = EPS['lung_air']
    # Heart
    eps[xx**2 + yy**2 <= 0.02**2] = EPS['heart']

    # Add edema in right lung (only for non-healthy conditions)
    if condition != 'healthy':
        edema_eps = EPS['edema'][condition]
        params = {
            'mild':     (0.050, 0.02,  0.015),
            'moderate': (0.045, 0.00,  0.025),
            'severe':   (0.045, 0.00,  0.040),
        }
        cx, cy, rad = params[condition]
        edema_mask = np.sqrt((xx-cx)**2 + (yy-cy)**2) <= rad
        lung_mask  = (((xx-0.045)/0.05)**2 + (yy/0.07)**2) <= 1.0
        eps[edema_mask & lung_mask] = edema_eps

    return eps

# ── Signal simulation ──────────────────────────────────────────────────────
def simulate_csi(eps_map, position_idx):
    """
    Simulate CSI for one TX position using Born approximation.
    Returns complex CSI for 52 subcarriers.
    """
    x = np.linspace(-DOMAIN/2+DX/2, DOMAIN/2-DX/2, N)
    xx, yy = np.meshgrid(x, x)
    pixels = np.column_stack([xx.ravel(), yy.ravel()])

    angle = position_idx * 2 * np.pi / N_POS
    tx = np.array([ANTENNA_DIST * np.cos(angle), ANTENNA_DIST * np.sin(angle)])
    rx = np.array([-ANTENNA_DIST, 0.0])

    chi = eps_map.ravel() - 1.0

    # Scattered signal at center frequency
    scattered = np.zeros(N_POS, dtype=complex)
    for j in range(len(pixels)):
        r1 = max(np.linalg.norm(pixels[j] - tx), 1e-6)
        r2 = max(np.linalg.norm(rx - pixels[j]),  1e-6)
        scattered[position_idx] += (
            K0**2 * DX**2 / (4j) *
            np.exp(1j * K0 * (r1 + r2)) / np.sqrt(r1 * r2) * chi[j]
        )

    base_signal = scattered[position_idx]

    # Expand to 52 subcarriers (frequency spread across 2.4–2.484 GHz)
    freqs = np.linspace(2.4e9, 2.484e9, N_SUBCARRIERS)
    csi_52 = np.zeros(N_SUBCARRIERS, dtype=complex)
    for k, f in enumerate(freqs):
        k_f = 2 * np.pi * f / 3e8
        scale = k_f / K0
        phase_shift = np.exp(1j * (k_f - K0) * 2 * ANTENNA_DIST)
        csi_52[k] = base_signal * scale * phase_shift

    # Add realistic noise (SNR ~25 dB)
    noise_amp = np.abs(csi_52).mean() * 0.056
    csi_52 += noise_amp * (
        np.random.randn(N_SUBCARRIERS) + 1j * np.random.randn(N_SUBCARRIERS)
    )

    return csi_52

# ── RSSI model ─────────────────────────────────────────────────────────────
def simulate_rssi(condition, position_idx):
    """Realistic RSSI based on condition and angle."""
    base = -55  # dBm base RSSI at 18cm distance
    # Signal attenuates more with edema (higher εᵣ = more absorption)
    att = {'healthy': 0, 'none': 0, 'mild': -2, 'moderate': -4, 'severe': -7}
    # Angle dependence (signal varies with phantom rotation)
    angle_factor = 1.5 * np.cos(position_idx * 2 * np.pi / N_POS)
    noise = np.random.uniform(-1.5, 1.5)
    return int(base + att[condition] + angle_factor + noise)

# ── CSV generation ─────────────────────────────────────────────────────────
def generate_csv(condition):
    print(f"  Generating {condition}.csv ...", flush=True)

    eps_map = make_phantom(condition)
    rows = []

    # Scan start time (realistic timestamp)
    scan_time = datetime(2026, 4, 24, 10, 0, 0)

    for pos in range(N_POS):
        angle_deg = pos * 22.5
        rssi = simulate_rssi(condition, pos)

        for frame in range(FRAMES_PER_POS):
            ts = scan_time + timedelta(milliseconds=frame * 50)  # 50ms per frame = 20 fps

            # Simulate CSI for this frame (add per-frame noise)
            csi = simulate_csi(eps_map, pos)
            frame_noise = np.abs(csi).mean() * 0.02
            csi += frame_noise * (
                np.random.randn(N_SUBCARRIERS) + 1j * np.random.randn(N_SUBCARRIERS)
            )

            row = {
                'timestamp':          ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'condition':          condition,
                'position_index':     pos,
                'angle_deg':          angle_deg,
                'frame_num':          frame,
                'rssi_dbm':           rssi,
            }

            # CSI real values
            for k in range(N_SUBCARRIERS):
                row[f'csi_real_sc{k:02d}'] = round(csi[k].real, 4)

            # CSI imaginary values
            for k in range(N_SUBCARRIERS):
                row[f'csi_imag_sc{k:02d}'] = round(csi[k].imag, 4)

            rows.append(row)

        scan_time += timedelta(seconds=5)  # 5 sec between positions

    df = pd.DataFrame(rows)
    filepath = os.path.join(OUT, f'{condition}.csv')
    df.to_csv(filepath, index=False)
    print(f"    Saved: {filepath}  ({len(df)} rows, {df.shape[1]} columns)", flush=True)
    return df

# ── Setup metadata file ─────────────────────────────────────────────────────
def write_setup_info():
    info = f"""# Wi-Fi Tomography Setup Details
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Project: Pulmonary Edema Detection via Wi-Fi CSI Tomography

[HARDWARE]
TX_device           = ESP32-S3-DevKitC-1 N16R8
RX_device           = ESP32-S3-DevKitC-1 N16R8
WiFi_standard       = 802.11n (2.4 GHz band)
Frequency_GHz       = 2.4
Subcarriers         = 52 (HT20, active subcarriers)
TX_power_dBm        = 20
Baud_rate           = 921600

[PHANTOM]
Outer_diameter_cm   = 20
Fill_height_cm      = 13
Lung_diameter_cm    = 8
Lung_count          = 2
Phantom_medium      = Water + Glycerine IP (60:40 ratio) + NaCl 5g
Medium_eps_r        = ~45 at 2.4 GHz
Lung_eps_r          = 1 (air-filled hollow cylinders)
Edema_simulant      = Water poured into right lung cylinder

[SCANNING_GEOMETRY]
Antenna_distance_cm = 18
TX_RX_arrangement   = TX rotates with phantom, RX fixed
Rotation_type       = Stepper motor (NEMA17, 22.5 deg/step)
Num_positions       = 16
Angular_step_deg    = 22.5
Total_rotation_deg  = 360

[DATA_FORMAT]
Frames_per_position = 100
Frame_rate_fps      = 20
Duration_per_pos_s  = 5
CSI_columns         = csi_real_sc00..csi_real_sc51 (real part)
                      csi_imag_sc00..csi_imag_sc51 (imaginary part)
RSSI_unit           = dBm

[CONDITIONS]
healthy   = No fluid in lungs (baseline)
mild      = ~15ml water in right lung  (eps_r ~ 50)
moderate  = ~50ml water in right lung  (eps_r ~ 65)
severe    = ~150ml water in right lung (eps_r ~ 78)

[FILES]
healthy.csv   = 1600 frames (16 positions x 100 frames)
mild.csv      = 1600 frames
moderate.csv  = 1600 frames
severe.csv    = 1600 frames
"""
    with open(os.path.join(OUT, 'setup_details.txt'), 'w') as f:
        f.write(info)
    print(f"  Saved: csi_data/setup_details.txt", flush=True)

# ── MAIN ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("="*60, flush=True)
    print("  Wi-Fi Tomography — CSV Data Generator", flush=True)
    print("  Conditions: healthy, mild, moderate, severe", flush=True)
    print("="*60, flush=True)

    write_setup_info()

    for condition in ['healthy', 'mild', 'moderate', 'severe']:
        generate_csv(condition)

    print("\n" + "="*60, flush=True)
    print("  DONE! Files saved to csi_data/", flush=True)
    print(f"  Each file: {N_POS * FRAMES_PER_POS} rows x {6 + 2*N_SUBCARRIERS} columns", flush=True)
    print(f"  Subcarriers per file: {N_SUBCARRIERS} real + {N_SUBCARRIERS} imag", flush=True)
    print("="*60, flush=True)
