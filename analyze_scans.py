"""
Full analysis of real scan data - honest assessment.
"""
import json, numpy as np

R = json.load(open('dashboard/real_scan_data.json'))

print("FULL ANALYSIS OF YOUR REAL ESP32 SCAN DATA")
print("="*60)

for c in ['healthy','mild','moderate','severe']:
    s = R['scans'][c]
    amps = [p['amp_mean'] for p in s['positions']]
    rssi = [p['rssi_dbm'] for p in s['positions']]
    print(f"\n{c.upper()}")
    print(f"  RSSI: {np.mean(rssi):.1f} dBm | Amp: {np.mean(amps):.4f}")

h = [p['amp_mean'] for p in R['scans']['healthy']['positions']]

print("\n\nDIFFERENCES vs HEALTHY:")
for c in ['mild','moderate','severe']:
    ca = [p['amp_mean'] for p in R['scans'][c]['positions']]
    d = [ca[i]-h[i] for i in range(16)]
    ad = [abs(x) for x in d]
    print(f"  {c}: mean_diff={np.mean(ad):.4f}, max_diff={max(ad):.4f} at pos {ad.index(max(ad))}")

m = [p['amp_mean'] for p in R['scans']['moderate']['positions']]
s = [p['amp_mean'] for p in R['scans']['severe']['positions']]
d = [s[i]-m[i] for i in range(16)]
print(f"\n  severe-moderate: mean_diff={np.mean([abs(x) for x in d]):.4f}")
