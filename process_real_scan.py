"""Process real scan data and export to dashboard JSON."""
import numpy as np, json, os, glob

scan_dir = 'real_scans'
output = 'dashboard/real_scan_data.json'

# Find all scan files
scans = {}
for f in sorted(glob.glob(os.path.join(scan_dir, '*_summary.json'))):
    s = json.load(open(f))
    cond = s['scan_info']['condition']
    
    # Load corresponding npz
    npz_path = f.replace('_summary.json', '.npz')
    if not os.path.exists(npz_path):
        continue
    d = np.load(npz_path)
    
    # Extract CSI data per position
    positions = []
    for pos in range(16):
        key_real = f'pos{pos:02d}_csi_real'
        key_imag = f'pos{pos:02d}_csi_imag'
        key_amp = f'pos{pos:02d}_amplitude'
        key_rssi = f'pos{pos:02d}_rssi'
        
        if key_real in d and key_imag in d:
            csi_real = d[key_real].tolist()
            csi_imag = d[key_imag].tolist()
            amp = d[key_amp].tolist()
            rssi = float(d[key_rssi][0])
            positions.append({
                'position': pos,
                'angle': pos * 22.5,
                'csi_real': csi_real[:64],  # Keep first 64 subcarriers
                'csi_imag': csi_imag[:64],
                'amplitude': amp[:64],
                'rssi_dbm': rssi,
                'amp_mean': float(np.mean(amp)),
                'amp_max': float(np.max(amp)),
                'n_subcarriers': len(csi_real)
            })
    
    # Per-position summary from JSON
    per_pos = s.get('per_position', {})
    
    scans[cond] = {
        'scan_info': s['scan_info'],
        'hardware': s['hardware'],
        'quality': s['quality_metrics'],
        'positions': positions,
        'summary': {
            'n_positions': len(positions),
            'mean_rssi': round(np.mean([p['rssi_dbm'] for p in positions]), 1),
            'mean_amp': round(np.mean([p['amp_mean'] for p in positions]), 4),
            'max_amp': round(max([p['amp_max'] for p in positions]), 4),
            'n_subcarriers': positions[0]['n_subcarriers'] if positions else 0,
        },
        'csi_avg_amplitude': np.mean([p['amplitude'] for p in positions], axis=0).tolist() if positions else [],
        'file': os.path.basename(npz_path)
    }
    
    print(f"Processed: {cond}")
    print(f"  Positions: {len(positions)}")
    print(f"  RSSI range: {min(p['rssi_dbm'] for p in positions):.1f} to {max(p['rssi_dbm'] for p in positions):.1f} dBm")
    print(f"  Amp range: {min(p['amp_mean'] for p in positions):.4f} to {max(p['amp_mean'] for p in positions):.4f}")

with open(output, 'w') as f:
    json.dump({'scans': scans, 'conditions_done': list(scans.keys())}, f, indent=2)

print(f"\nExported {len(scans)} scans to {output}")
print(f"Conditions: {list(scans.keys())}")
