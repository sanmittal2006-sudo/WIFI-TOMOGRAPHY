import json, numpy as np
d = json.load(open('dashboard/meep_data.json'))
samples = d['all_samples']
print(f"Total samples: {len(samples)}")

# Check structure of first sample
s0 = samples[0]
print(f"Keys: {list(s0.keys())}")
print(f"Severity: {s0.get('severity','?')}, edema_level: {s0.get('edema_level','?')}")

# Check if all CSI data is the same
for sev in ['none','mild','moderate','severe']:
    sev_samples = [s for s in samples if s.get('edema_level','') == sev or s.get('severity','') == sev]
    if not sev_samples:
        sev_samples = [s for s in samples if sev in str(s.get('severity','')).lower() or sev in str(s.get('edema_level','')).lower()]
    if sev_samples:
        print(f"\n{sev.upper()} ({len(sev_samples)} samples):")
        for i, s in enumerate(sev_samples[:2]):
            csi = s.get('csi_differential', s.get('csi_data', []))
            peak = s.get('peak_delta_e', 0)
            print(f"  S{i}: peak={peak:.4f}, csi[0:3]={[round(x,4) for x in csi[:3]]}, max={max(csi):.4f}, min={min(csi):.4f}")
    else:
        print(f"\n{sev}: NO SAMPLES FOUND")

# Check all peak values
peaks = [s.get('peak_delta_e', 0) for s in samples]
print(f"\nAll peaks: min={min(peaks):.4f}, max={max(peaks):.4f}")
print(f"Unique peaks (rounded): {len(set([round(p,3) for p in peaks]))}")
