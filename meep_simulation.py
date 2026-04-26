#!/usr/bin/env python3
"""
MEEP FDTD Simulation for Wi-Fi Microwave Tomography
====================================================
This script uses MEEP (MIT Electromagnetic Equation Propagation)
to simulate Wi-Fi signal propagation through a chest phantom.

MEEP solves Maxwell's equations using FDTD (Finite Difference Time Domain).
This is the GOLD STANDARD for electromagnetic simulation.

Usage:
  Run inside WSL Ubuntu:
    conda activate meep
    python meep_simulation.py

Output:
  meep_training_data/  folder with simulated CSI data
  
Author: Wi-Fi Tomography Project Team
"""

import meep as mp
import numpy as np
import os
import time
import json

# ============================================================================
#   CONFIGURATION
# ============================================================================

FREQ_GHZ = 2.4               # Wi-Fi frequency
C = 3e8                       # Speed of light (m/s)
WAVELENGTH = C / (FREQ_GHZ * 1e9)  # ~0.125 m = 12.5 cm

# MEEP uses normalized units where c = 1
# We set a = 1 meter (our length unit)
# So frequency in MEEP = physical_freq * a / c = 2.4e9 / 3e8 = 8.0
FREQ_MEEP = FREQ_GHZ * 1e9 / C  # = 8.0 (in MEEP units, with a=1m)

# Domain
DOMAIN_SIZE_CM = 30.0         # 30 cm × 30 cm imaging area
DOMAIN_SIZE_M = DOMAIN_SIZE_CM / 100.0
ANTENNA_RADIUS_CM = 18.0      # Antennas at 18 cm from center
ANTENNA_RADIUS_M = ANTENNA_RADIUS_CM / 100.0
NUM_POSITIONS = 16            # 16 rotation angles (22.5° each)
RESOLUTION = 30               # MEEP pixels per meter (higher = more accurate)

# Dielectric properties at 2.4 GHz
TISSUE_PROPERTIES = {
    'air':    1.0,
    'skin':   38.0,
    'fat':    5.3,
    'muscle': 52.7,
    'bone':   13.1,
    'lung_healthy': 3.0,      # Air-filled healthy lung
    'lung_edema':   68.0,     # Fluid-filled edematous lung  
    'heart':  58.0,
    'blood':  65.0,
    'agar':   52.0,           # Agar gel phantom
    'water':  78.0,           # Pure water (edema simulant)
}

# ============================================================================
#   PHANTOM CREATION
# ============================================================================

def create_meep_phantom(phantom_type='agar_simple', edema_level='none',
                        domain_size=DOMAIN_SIZE_M, resolution=RESOLUTION):
    """
    Create a MEEP geometry for a chest phantom.
    
    phantom_type: 'agar_simple' or 'chest_anatomical'
    edema_level: 'none', 'mild', 'moderate', 'severe'
    
    Returns:
        geometry: List of MEEP geometry objects
        eps_map: 2D numpy array of dielectric values (for ground truth)
    """
    geometry = []
    
    if phantom_type == 'agar_simple':
        # ============================================================
        # AGAR PHANTOM (what you actually build in lab)
        # 20 cm diameter cylinder filled with agar gel (εr=52)
        # ============================================================
        
        # Agar cylinder (20cm diameter = 10cm radius)
        agar_radius = 0.10  # 10 cm in meters
        geometry.append(mp.Cylinder(
            radius=agar_radius,
            height=mp.inf,
            center=mp.Vector3(0, 0),
            material=mp.Medium(epsilon=TISSUE_PROPERTIES['agar'])
        ))
        
        if edema_level != 'none':
            # Add water-filled region to simulate edema
            edema_params = {
                'mild':     {'radius': 0.02, 'pos': (0.04, 0)},   # 2cm, offset 4cm
                'moderate': {'radius': 0.03, 'pos': (0.03, 0)},   # 3cm, offset 3cm
                'severe':   {'radius': 0.05, 'pos': (0.03, 0)},   # 5cm, offset 3cm
            }
            p = edema_params[edema_level]
            geometry.append(mp.Cylinder(
                radius=p['radius'],
                height=mp.inf,
                center=mp.Vector3(p['pos'][0], p['pos'][1]),
                material=mp.Medium(epsilon=TISSUE_PROPERTIES['water'])
            ))
    
    elif phantom_type == 'chest_anatomical':
        # ============================================================
        # ANATOMICAL CHEST (for simulation/paper only)
        # Elliptical cross-section with organs
        # ============================================================
        
        # Chest wall (ellipse: 28cm × 22cm)
        geometry.append(mp.Ellipsoid(
            size=mp.Vector3(0.28, 0.22),
            center=mp.Vector3(0, 0),
            material=mp.Medium(epsilon=TISSUE_PROPERTIES['muscle'])
        ))
        
        # Fat layer (slightly smaller ellipse)
        geometry.append(mp.Ellipsoid(
            size=mp.Vector3(0.26, 0.20),
            center=mp.Vector3(0, 0),
            material=mp.Medium(epsilon=TISSUE_PROPERTIES['fat'])
        ))
        
        # Left lung
        geometry.append(mp.Ellipsoid(
            size=mp.Vector3(0.08, 0.14),
            center=mp.Vector3(-0.06, 0.01),
            material=mp.Medium(epsilon=TISSUE_PROPERTIES['lung_healthy'])
        ))
        
        # Right lung (may have edema)
        right_lung_er = TISSUE_PROPERTIES['lung_healthy']
        if edema_level == 'mild':
            right_lung_er = 20.0
        elif edema_level == 'moderate':
            right_lung_er = 45.0
        elif edema_level == 'severe':
            right_lung_er = TISSUE_PROPERTIES['lung_edema']
        
        geometry.append(mp.Ellipsoid(
            size=mp.Vector3(0.08, 0.14),
            center=mp.Vector3(0.06, 0.01),
            material=mp.Medium(epsilon=right_lung_er)
        ))
        
        # Heart
        geometry.append(mp.Cylinder(
            radius=0.035,
            height=mp.inf,
            center=mp.Vector3(-0.02, 0.02),
            material=mp.Medium(epsilon=TISSUE_PROPERTIES['heart'])
        ))
    
    return geometry


def run_single_meep_simulation(geometry, tx_angle_deg, rx_angle_deg,
                                domain_size=DOMAIN_SIZE_M,
                                antenna_radius=ANTENNA_RADIUS_M,
                                freq=FREQ_MEEP,
                                resolution=RESOLUTION):
    """
    Run a single MEEP FDTD simulation with TX at one angle and RX at another.
    
    Returns:
        complex_field: Complex E-field amplitude at the RX position
    """
    # Convert angles to positions
    tx_rad = np.radians(tx_angle_deg)
    rx_rad = np.radians(rx_angle_deg)
    
    tx_pos = mp.Vector3(antenna_radius * np.cos(tx_rad),
                        antenna_radius * np.sin(tx_rad))
    rx_pos = mp.Vector3(antenna_radius * np.cos(rx_rad),
                        antenna_radius * np.sin(rx_rad))
    
    # MEEP source: continuous wave at 2.4 GHz
    sources = [mp.Source(
        mp.ContinuousSource(frequency=freq),
        component=mp.Ez,  # Vertical polarization (matches ESP32 orientation)
        center=tx_pos,
        size=mp.Vector3(0, 0),
    )]
    
    # Simulation cell — must be large enough that antennas are NOT inside PML!
    # Antenna radius = 0.18m, so cell must be > 2*(0.18 + PML_thickness)
    cell_dim = 2 * (antenna_radius + 0.07)  # 0.50m for 0.18m antennas
    cell_size = mp.Vector3(cell_dim, cell_dim)
    
    # PML absorbing boundary (prevents reflections from edges)
    pml_thickness = 0.05  # 5cm PML layer
    pml = [mp.PML(thickness=pml_thickness)]
    
    # Create simulation
    sim = mp.Simulation(
        cell_size=cell_size,
        boundary_layers=pml,
        geometry=geometry,
        sources=sources,
        resolution=resolution,
        force_complex_fields=True,
    )
    
    # Run until steady state
    # CW source needs many oscillation periods to build up
    # 1 period = 1/freq_meep. We need ~200 periods for steady state.
    num_periods = 200
    run_time = num_periods / freq
    sim.run(until=run_time)
    
    # Get complex field at RX position
    ez = sim.get_field_point(mp.Ez, rx_pos)
    
    # Clean up
    sim.reset_meep()
    
    return complex(ez)


def run_full_meep_scan(geometry, num_positions=NUM_POSITIONS):
    """
    Run full 16-position MEEP scan.
    TX is FIXED at 0°, phantom rotates (equivalent to rotating TX).
    
    Returns:
        csi_vector: Complex numpy array of shape (num_positions,)
    """
    csi = np.zeros(num_positions, dtype=complex)
    
    for i in range(num_positions):
        tx_angle = i * (360.0 / num_positions)  # TX rotates
        rx_angle = 180.0  # RX fixed opposite
        
        print(f"    Position {i+1}/{num_positions}: TX={tx_angle:.1f}° → RX={rx_angle:.1f}°", 
              end="", flush=True)
        
        t0 = time.time()
        csi[i] = run_single_meep_simulation(geometry, tx_angle, rx_angle)
        dt = time.time() - t0
        
        print(f"  |Ez|={np.abs(csi[i]):.6f}  phase={np.angle(csi[i]):.2f}  ({dt:.1f}s)")
    
    return csi


# ============================================================================
#   TRAINING DATA GENERATION
# ============================================================================

def generate_meep_training_dataset(num_samples=2, phantom_type='agar_simple'):
    """
    Generate MEEP-simulated training data for U-Net.
    
    Each sample:
      1. Random phantom with random 'edema' region
      2. Full MEEP FDTD scan → CSI measurements
      3. Save both ground truth εr map and CSI
    
    Args:
        num_samples: Number of training pairs to generate
        phantom_type: 'agar_simple' or 'chest_anatomical'
    """
    output_dir = 'meep_training_data'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"  MEEP FDTD TRAINING DATA GENERATION")
    print(f"  Samples: {num_samples}")
    print(f"  Phantom: {phantom_type}")
    print(f"  Frequency: {FREQ_GHZ} GHz (λ = {WAVELENGTH*100:.1f} cm)")
    print(f"  Resolution: {RESOLUTION} pixels/m")
    print(f"{'='*60}\n")
    
    all_results = []
    
    for sample_idx in range(num_samples):
        print(f"\n--- Sample {sample_idx + 1}/{num_samples} ---")
        
        # Randomly choose edema level
        edema_levels = ['none', 'mild', 'moderate', 'severe']
        edema = np.random.choice(edema_levels)
        print(f"  Edema level: {edema}")
        
        # Create geometry
        geometry = create_meep_phantom(phantom_type, edema)
        
        # Run empty scan (baseline - no phantom)
        print(f"\n  Running EMPTY baseline scan...")
        t0 = time.time()
        csi_empty = run_full_meep_scan([], num_positions=NUM_POSITIONS)
        t_empty = time.time() - t0
        print(f"  Empty scan complete ({t_empty:.1f}s)")
        
        # Run phantom scan
        print(f"\n  Running PHANTOM scan (edema={edema})...")
        t0 = time.time()
        csi_phantom = run_full_meep_scan(geometry, num_positions=NUM_POSITIONS)
        t_phantom = time.time() - t0
        print(f"  Phantom scan complete ({t_phantom:.1f}s)")
        
        # Differential CSI (this is what BIM uses)
        csi_diff = csi_phantom - csi_empty
        
        # Save result
        result = {
            'sample_idx': sample_idx,
            'edema_level': edema,
            'phantom_type': phantom_type,
            'csi_empty': csi_empty,
            'csi_phantom': csi_phantom, 
            'csi_differential': csi_diff,
            'time_empty_sec': t_empty,
            'time_phantom_sec': t_phantom,
        }
        
        # Save as .npz file
        filename = f'{output_dir}/meep_sample_{sample_idx:04d}_{edema}.npz'
        np.savez(filename,
            csi_empty=csi_empty,
            csi_phantom=csi_phantom,
            csi_differential=csi_diff,
            edema_level=edema,
            phantom_type=phantom_type,
        )
        print(f"  💾 Saved: {filename}")
        
        all_results.append(result)
    
    # Save metadata
    metadata = {
        'num_samples': num_samples,
        'phantom_type': phantom_type,
        'freq_ghz': FREQ_GHZ,
        'num_positions': NUM_POSITIONS,
        'domain_size_cm': DOMAIN_SIZE_CM,
        'antenna_radius_cm': ANTENNA_RADIUS_CM,
        'resolution': RESOLUTION,
        'tissue_properties': TISSUE_PROPERTIES,
    }
    
    with open(f'{output_dir}/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"  ✅ MEEP TRAINING DATA COMPLETE")
    print(f"  {num_samples} samples saved to {output_dir}/")
    print(f"{'='*60}")
    
    return all_results


# ============================================================================
#   MAIN
# ============================================================================

if __name__ == '__main__':
    print("🌊 MEEP FDTD Electromagnetic Simulator")
    print(f"   Version: {mp.__version__}")
    print(f"   Frequency: {FREQ_GHZ} GHz")
    print(f"   Wavelength: {WAVELENGTH*100:.1f} cm")
    print()
    
    # Generate 2 MEEP samples first (as proof of concept)
    # Each takes ~5-10 minutes on your laptop
    # You can increase num_samples later
    results = generate_meep_training_dataset(
        num_samples=500,
        phantom_type='agar_simple'
    )
    
    print("\n📊 Results Summary:")
    for r in results:
        print(f"  Sample {r['sample_idx']}: edema={r['edema_level']}")
        print(f"    |CSI_diff| range: {np.min(np.abs(r['csi_differential'])):.6f} "
              f"to {np.max(np.abs(r['csi_differential'])):.6f}")
        print(f"    Time: {r['time_empty_sec'] + r['time_phantom_sec']:.1f}s total")
