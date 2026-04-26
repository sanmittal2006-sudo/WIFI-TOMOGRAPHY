#!/usr/bin/env python3
"""
Quick Reconstruction — Visualize real scan data as heatmap
Shows if we can see the lungs inside the phantom
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Load the real scan
scan_file = 'real_scans/healthy_20260425_044540.npz'
data = np.load(scan_file)

print("Loaded scan file:", scan_file)
print("Keys:", list(data.keys())[:10], "...")

# ── Extract CSI amplitudes for all 16 positions ──────────
N_POS = 16
amplitudes = []
phases = []

for pos in range(N_POS):
    amp_key = f'pos{pos:02d}_amplitude'
    phase_key = f'pos{pos:02d}_phase'
    
    if amp_key in data:
        amplitudes.append(data[amp_key])
        phases.append(data[phase_key])
        print(f"  Pos {pos:2d} ({pos*22.5:5.1f}°): {len(data[amp_key])} subcarriers, "
              f"amp_mean={np.mean(data[amp_key]):.3f}")

print(f"\nTotal positions loaded: {len(amplitudes)}")

# ── Normalize to common length ────────────────────────────
min_len = min(len(a) for a in amplitudes)
amp_matrix = np.array([a[:min_len] for a in amplitudes])   # [16, N_sub]
phase_matrix = np.array([p[:min_len] for p in phases])

print(f"Matrix shape: {amp_matrix.shape}")

# ── Build sinogram (amplitude vs angle vs subcarrier) ─────
# This is like a CT sinogram — each row is one angle
sinogram = amp_matrix  # [16 angles, N subcarriers]

# ── Simple backprojection reconstruction ──────────────────
# Same principle as CT: project each angle's data back
N_GRID = 64  # 64x64 pixel reconstruction
recon = np.zeros((N_GRID, N_GRID))

# Domain: 20cm phantom → ±10cm
x = np.linspace(-0.10, 0.10, N_GRID)
y = np.linspace(-0.10, 0.10, N_GRID)
X, Y = np.meshgrid(x, y)

# Circular mask (phantom boundary)
R = np.sqrt(X**2 + Y**2)
mask = R <= 0.10  # 10cm radius

for pos in range(N_POS):
    angle_rad = pos * 22.5 * np.pi / 180.0
    
    # Project direction
    proj_x = np.cos(angle_rad)
    proj_y = np.sin(angle_rad)
    
    # Project each pixel onto the measurement line
    projection = X * proj_x + Y * proj_y
    
    # Normalize projection to subcarrier index
    proj_norm = (projection - projection.min()) / (projection.max() - projection.min() + 1e-10)
    proj_idx = (proj_norm * (min_len - 1)).astype(int)
    proj_idx = np.clip(proj_idx, 0, min_len - 1)
    
    # Backproject amplitude
    recon += amp_matrix[pos][proj_idx]

# Apply circular mask
recon *= mask

# Normalize
recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-10)

# ── Also do phase-based reconstruction ────────────────────
recon_phase = np.zeros((N_GRID, N_GRID))

for pos in range(N_POS):
    angle_rad = pos * 22.5 * np.pi / 180.0
    proj_x = np.cos(angle_rad)
    proj_y = np.sin(angle_rad)
    projection = X * proj_x + Y * proj_y
    proj_norm = (projection - projection.min()) / (projection.max() - projection.min() + 1e-10)
    proj_idx = (proj_norm * (min_len - 1)).astype(int)
    proj_idx = np.clip(proj_idx, 0, min_len - 1)
    recon_phase += np.abs(phase_matrix[pos][proj_idx])

recon_phase *= mask
recon_phase = (recon_phase - recon_phase.min()) / (recon_phase.max() - recon_phase.min() + 1e-10)

# ── Combined reconstruction ───────────────────────────────
recon_combined = 0.6 * recon + 0.4 * recon_phase
recon_combined = (recon_combined - recon_combined.min()) / (recon_combined.max() - recon_combined.min() + 1e-10)

# ── Plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Sinogram
im0 = axes[0,0].imshow(sinogram, aspect='auto', cmap='viridis',
                         extent=[0, min_len, 337.5, 0])
axes[0,0].set_xlabel('Subcarrier Index')
axes[0,0].set_ylabel('Angle (degrees)')
axes[0,0].set_title('CSI Sinogram (Amplitude)', fontsize=13, fontweight='bold')
plt.colorbar(im0, ax=axes[0,0], label='Amplitude')

# 2. Amplitude per position (polar)
angles = np.array([i * 22.5 for i in range(N_POS)])
mean_amps = np.mean(amp_matrix, axis=1)
ax_polar = fig.add_subplot(2, 3, 2, polar=True)
axes[0,1].remove()
ax_polar.set_position(axes[0,1].get_position())
ax_polar.bar(np.radians(angles), mean_amps, width=np.radians(20), alpha=0.7, color='coral')
ax_polar.set_title('Mean Amplitude per Angle', fontsize=13, fontweight='bold', pad=20)

# 3. Amplitude heatmap reconstruction
im2 = axes[0,2].imshow(recon, cmap='hot', extent=[-10, 10, -10, 10])
axes[0,2].set_xlabel('X (cm)')
axes[0,2].set_ylabel('Y (cm)')
axes[0,2].set_title('Backprojection (Amplitude)', fontsize=13, fontweight='bold')
# Draw expected lung positions
lung_r = 0.04  # 4cm radius lungs
circle1 = plt.Circle((-0.03*100, 0), lung_r*100, fill=False, color='cyan', linewidth=2, linestyle='--', label='Expected lungs')
circle2 = plt.Circle((0.03*100, 0), lung_r*100, fill=False, color='cyan', linewidth=2, linestyle='--')
axes[0,2].add_patch(circle1)
axes[0,2].add_patch(circle2)
axes[0,2].legend(loc='upper right', fontsize=9)
plt.colorbar(im2, ax=axes[0,2])

# 4. Phase reconstruction
im3 = axes[1,0].imshow(recon_phase, cmap='hot', extent=[-10, 10, -10, 10])
axes[1,0].set_xlabel('X (cm)')
axes[1,0].set_ylabel('Y (cm)')
axes[1,0].set_title('Backprojection (Phase)', fontsize=13, fontweight='bold')
plt.colorbar(im3, ax=axes[1,0])

# 5. Combined reconstruction
im4 = axes[1,1].imshow(recon_combined, cmap='inferno', extent=[-10, 10, -10, 10])
axes[1,1].set_xlabel('X (cm)')
axes[1,1].set_ylabel('Y (cm)')
axes[1,1].set_title('Combined Reconstruction', fontsize=13, fontweight='bold')
circle3 = plt.Circle((-3, 0), 4, fill=False, color='lime', linewidth=2, linestyle='--')
circle4 = plt.Circle((3, 0), 4, fill=False, color='lime', linewidth=2, linestyle='--')
axes[1,1].add_patch(circle3)
axes[1,1].add_patch(circle4)
plt.colorbar(im4, ax=axes[1,1])

# 6. CSI amplitude profiles
axes[1,2].plot(amp_matrix[0], label='0°', alpha=0.8)
axes[1,2].plot(amp_matrix[4], label='90°', alpha=0.8)
axes[1,2].plot(amp_matrix[8], label='180°', alpha=0.8)
axes[1,2].plot(amp_matrix[12], label='270°', alpha=0.8)
axes[1,2].set_xlabel('Subcarrier Index')
axes[1,2].set_ylabel('CSI Amplitude')
axes[1,2].set_title('CSI Profiles at Key Angles', fontsize=13, fontweight='bold')
axes[1,2].legend()
axes[1,2].grid(True, alpha=0.3)

plt.suptitle('Wi-Fi Tomography — Real Phantom Scan (HEALTHY)\n'
             'Glycerine container with 2 empty air lungs',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/healthy_reconstruction.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: results/healthy_reconstruction.png")
plt.show()
