"""Extract all real MEEP data and export as JSON for the dashboard."""
import numpy as np, json, glob, os

files = sorted(glob.glob('meep_training_data/*.npz'))
metadata = json.load(open('meep_training_data/metadata.json'))

levels = {'none': [], 'mild': [], 'moderate': [], 'severe': []}

for f in files:
    d = np.load(f, allow_pickle=True)
    l = str(d['edema_level'])
    levels[l].append({
        'csi_empty': np.abs(d['csi_empty']).tolist(),
        'csi_phantom': np.abs(d['csi_phantom']).tolist(),
        'csi_differential': np.abs(d['csi_differential']).tolist(),
        'csi_diff_real': d['csi_differential'].real.tolist(),
        'csi_diff_imag': d['csi_differential'].imag.tolist(),
        'file': os.path.basename(f)
    })

# Average CSI per level
avg_csi = {}
for l, samples in levels.items():
    if samples:
        all_diff = [s['csi_differential'] for s in samples]
        avg_csi[l] = {
            'mean': np.mean(all_diff, axis=0).tolist(),
            'std': np.std(all_diff, axis=0).tolist(),
            'min': np.min(all_diff, axis=0).tolist(),
            'max': np.max(all_diff, axis=0).tolist(),
            'count': len(samples)
        }

# BIM config
bim_config = {
    'grid_size': 32,
    'domain_m': 0.30,
    'antenna_radius_m': 0.18,
    'n_positions': 16,
    'frequency_hz': 2.4e9,
    'wavelength_m': 3e8 / 2.4e9,
    'k0': 2 * 3.14159265 * 2.4e9 / 3e8,
    'dx': 0.30 / 32,
    'bim_iterations': 400,
    'bim_lambda': 0.001,
    'bim_relaxation': 0.5,
    'ssim': {'none': 0.295, 'mild': 0.282, 'moderate': 0.255, 'severe': 0.233}
}

# Phantom properties
phantom = {
    'none': {'edema_cx': 0, 'edema_cy': 0, 'edema_radius': 0, 'edema_er': 0, 'agar_er': 52.0},
    'mild': {'edema_cx': 0.04, 'edema_cy': 0, 'edema_radius': 0.02, 'edema_er': 65, 'agar_er': 52.0},
    'moderate': {'edema_cx': 0.03, 'edema_cy': 0, 'edema_radius': 0.03, 'edema_er': 72, 'agar_er': 52.0},
    'severe': {'edema_cx': 0.03, 'edema_cy': 0, 'edema_radius': 0.05, 'edema_er': 78, 'agar_er': 52.0}
}

export = {
    'metadata': metadata,
    'bim_config': bim_config,
    'phantom_properties': phantom,
    'average_csi': avg_csi,
    'all_samples': {l: samples for l, samples in levels.items()},
    'antenna_angles_deg': [i * 22.5 for i in range(16)]
}

with open('dashboard/meep_data.json', 'w') as f:
    json.dump(export, f, indent=2)

print(f"Exported {sum(len(v) for v in levels.values())} samples to dashboard/meep_data.json")
for l, s in levels.items():
    print(f"  {l}: {len(s)} samples")
print(f"\nMetadata: {json.dumps(metadata, indent=2)}")
