import numpy as np
import os

print("=" * 50)
print("  MEEP TRAINING DATA INSPECTION")
print("=" * 50)

data_dir = "meep_training_data"
for f in sorted(os.listdir(data_dir)):
    if f.endswith('.npz'):
        path = os.path.join(data_dir, f)
        d = np.load(path, allow_pickle=True)
        print(f"\nFile: {f}")
        print(f"  Edema level: {d['edema_level']}")
        print(f"  Phantom type: {d['phantom_type']}")
        print(f"  CSI empty shape: {d['csi_empty'].shape}")
        print(f"  CSI phantom shape: {d['csi_phantom'].shape}")
        print(f"  CSI differential shape: {d['csi_differential'].shape}")
        print(f"  |CSI_empty| = {np.abs(d['csi_empty'])}")
        print(f"  |CSI_phantom| = {np.abs(d['csi_phantom'])}")
        print(f"  |CSI_diff| = {np.abs(d['csi_differential'])}")
        print(f"  Max diff: {np.max(np.abs(d['csi_differential'])):.4f}")
        print(f"  Min diff: {np.min(np.abs(d['csi_differential'])):.4f}")
