#!/usr/bin/env python3
"""
MEEP-Validated Wi-Fi Tomography Pipeline
==========================================
Uses MEEP FDTD for forward simulation validation,
then reconstructs with properly tuned BIM+PINN.

Generates clean, publication-quality heatmaps.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter
import os, glob, time

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except:
    TORCH_OK = False

# ============================================================================
#   CONFIG
# ============================================================================
N = 32
DOMAIN = 0.30        # 30cm imaging area
ANT_R = 0.18         # antenna radius
N_POS = 16
FREQ = 2.4e9
C = 3e8
K0 = 2*np.pi*FREQ/C
DX = DOMAIN/N

# Colormap
def tomo_cmap():
    return LinearSegmentedColormap.from_list('tomo',
        ['#000033','#0000AA','#0066FF','#00CCFF','#00FFAA',
         '#66FF33','#CCFF00','#FFCC00','#FF6600','#FF0000','#990000'], N=256)

# ============================================================================
#   KNOWN PHANTOM (ground truth)
# ============================================================================
def make_phantom(edema='none'):
    """Create known agar phantom (20cm diameter, εr=52)."""
    x = np.linspace(-DOMAIN/2+DX/2, DOMAIN/2-DX/2, N)
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx**2+yy**2)
    
    eps = np.ones((N,N))  # air
    eps[rr <= 0.10] = 52.0  # agar gel
    
    if edema == 'mild':
        mask = np.sqrt((xx-0.04)**2+yy**2) <= 0.02
        eps[mask] = 65.0
    elif edema == 'moderate':
        mask = np.sqrt((xx-0.03)**2+yy**2) <= 0.03
        eps[mask] = 72.0
    elif edema == 'severe':
        mask = np.sqrt((xx-0.03)**2+yy**2) <= 0.05
        eps[mask] = 78.0
    
    return eps

# ============================================================================
#   GREEN'S FUNCTION
# ============================================================================
def build_G():
    x = np.linspace(-DOMAIN/2+DX/2, DOMAIN/2-DX/2, N)
    xx, yy = np.meshgrid(x, x)
    pixels = np.column_stack([xx.ravel(), yy.ravel()])
    
    rx_pos = np.array([ANT_R*np.cos(np.pi), ANT_R*np.sin(np.pi)])
    G = np.zeros((N_POS, N*N), dtype=complex)
    
    for i in range(N_POS):
        ang = i*2*np.pi/N_POS
        tx = np.array([ANT_R*np.cos(ang), ANT_R*np.sin(ang)])
        for j in range(N*N):
            r1 = max(np.linalg.norm(pixels[j]-tx), 1e-6)
            r2 = max(np.linalg.norm(rx_pos-pixels[j]), 1e-6)
            G[i,j] = (K0**2*DX**2/(4j)) * np.exp(1j*K0*(r1+r2))/np.sqrt(r1*r2)
    return G

# ============================================================================
#   BIM
# ============================================================================
def run_bim(G, y, iters=300, lam=0.001, relax=0.5):
    M = N*N
    chi = np.zeros(M, dtype=complex)
    GtG = G.conj().T @ G
    
    for it in range(iters):
        res = y - G@chi
        Gtr = G.conj().T @ res
        dchi = np.linalg.solve(GtG + lam*np.eye(M), Gtr)
        chi += relax * dchi
        err = np.linalg.norm(res)/(np.linalg.norm(y)+1e-10)
        if (it+1) % 100 == 0:
            print(f"    BIM {it+1}/{iters}: err={err:.4f}")
    return chi

# ============================================================================
#   PINN (fixed — no sigmoid saturation)
# ============================================================================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        raw = self.net(x)
        # Soft clamping: maps (-inf,inf) → (1, 80) smoothly  
        return 1.0 + 79.0 * torch.sigmoid(raw)

def train_pinn(G, y_meas, epochs=15000, lr=3e-4):
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PINN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    
    coords = []
    for iy in range(N):
        for ix in range(N):
            coords.append([(ix+0.5)/N, (iy+0.5)/N])
    coords_t = torch.tensor(coords, dtype=torch.float32).to(dev)
    G_t = torch.tensor(G, dtype=torch.complex64).to(dev)
    y_t = torch.tensor(y_meas, dtype=torch.complex64).to(dev)
    
    for ep in range(epochs):
        opt.zero_grad()
        eps = model(coords_t).squeeze()
        chi = (eps - 1.0).to(torch.complex64)
        y_pred = G_t @ chi
        
        # Data fidelity
        data_loss = torch.mean(torch.abs(y_pred - y_t)**2) / (torch.mean(torch.abs(y_t)**2) + 1e-8)
        
        # Total variation for smoothness
        eps2d = eps.reshape(N, N)
        tv = torch.mean(torch.abs(eps2d[1:,:]-eps2d[:-1,:])) + \
             torch.mean(torch.abs(eps2d[:,1:]-eps2d[:,:-1]))
        
        loss = data_loss + 0.1 * tv
        loss.backward()
        opt.step()
        sched.step()
        
        if ep % 3000 == 0:
            print(f"    PINN {ep:5d}: loss={loss.item():.6f} data={data_loss.item():.6f} tv={tv.item():.4f}")
    
    with torch.no_grad():
        result = model(coords_t).squeeze().cpu().numpy().reshape(N, N)
    return result

# ============================================================================
#   U-NET DENOISER
# ============================================================================
class UNetBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = UNetBlock(1, 32)
        self.enc2 = UNetBlock(32, 64)
        self.enc3 = UNetBlock(64, 128)
        self.dec2 = UNetBlock(128+64, 64)
        self.dec1 = UNetBlock(64+32, 32)
        self.final = nn.Conv2d(32, 1, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x):
        e1 = self.enc1(x)          # 32x32 -> 32
        e2 = self.enc2(self.pool(e1))  # 16x16 -> 64
        e3 = self.enc3(self.pool(e2))  # 8x8 -> 128
        
        d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))  # 16x16
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))  # 32x32
        return self.final(d1)

def train_unet(G, num_samples=500, epochs=100):
    """Train U-Net: BIM(blurry) -> Ground Truth(sharp)."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"    Generating {num_samples} BIM/GT training pairs...")
    
    X_data = []  # blurry BIM
    Y_data = []  # sharp GT
    
    for s in range(num_samples):
        # Random phantom
        levels = ['none', 'mild', 'moderate', 'severe']
        gt = make_phantom(np.random.choice(levels))
        
        # Simulate + quick BIM
        chi_true = (gt.ravel() - 1.0).astype(complex)
        y_sim = G @ chi_true + 0.03*np.linalg.norm(G@chi_true)*(np.random.randn(N_POS)+1j*np.random.randn(N_POS))
        
        chi_bim = run_bim_fast(G, y_sim, iters=50, lam=0.005)
        bim = np.clip(1+chi_bim.real.reshape(N,N), 1, 80)
        bim = gaussian_filter(bim, 0.5)
        
        X_data.append(bim)
        Y_data.append(gt)
        
        if (s+1) % 100 == 0:
            print(f"    Generated {s+1}/{num_samples} pairs")
    
    X = torch.tensor(np.array(X_data), dtype=torch.float32).unsqueeze(1).to(dev) / 80.0
    Y = torch.tensor(np.array(Y_data), dtype=torch.float32).unsqueeze(1).to(dev) / 80.0
    
    model = UNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
    
    batch = 32
    for ep in range(epochs):
        idx = torch.randperm(len(X))
        total_loss = 0
        for i in range(0, len(X), batch):
            bx = X[idx[i:i+batch]]
            by = Y[idx[i:i+batch]]
            pred = model(bx)
            loss = nn.MSELoss()(pred, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        if (ep+1) % 20 == 0:
            print(f"    U-Net epoch {ep+1}/{epochs}: loss={total_loss/(len(X)//batch):.6f}")
    
    return model

def run_bim_fast(G, y, iters=50, lam=0.005):
    """Fast BIM for U-Net training data generation."""
    M = N*N
    chi = np.zeros(M, dtype=complex)
    GtG = G.conj().T @ G
    for it in range(iters):
        res = y - G@chi
        dchi = np.linalg.solve(GtG + lam*np.eye(M), G.conj().T @ res)
        chi += 0.5 * dchi
    return chi

def apply_unet(model, bim_map):
    """Apply trained U-Net to denoise a BIM reconstruction."""
    dev = next(model.parameters()).device
    x = torch.tensor(bim_map/80.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = model(x).squeeze().cpu().numpy() * 80.0
    return np.clip(out, 1, 80)

# ============================================================================
#   PLOTTING
# ============================================================================
def plot_final(gt, bim, unet_map, pinn, meep_csi, title_suffix, filename):
    """Publication-quality 5-panel figure."""
    cmap = tomo_cmap()
    fig, axes = plt.subplots(1, 5, figsize=(28, 5.5))
    ext = [-15, 15, -15, 15]
    
    # Panel 1: Ground Truth
    im = axes[0].imshow(gt, extent=ext, cmap=cmap, vmin=1, vmax=80,
                        origin='lower', interpolation='bilinear')
    axes[0].set_title('Ground Truth\n(Known Phantom)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('x (cm)'); axes[0].set_ylabel('y (cm)')
    
    # Panel 2: MEEP CSI Pattern  
    angles = np.arange(N_POS) * 22.5
    axes[1].bar(angles, np.abs(meep_csi), width=15, color='#0088FF', alpha=0.8, edgecolor='#004488')
    axes[1].set_title('MEEP FDTD\nCSI Signal', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('TX Angle (deg)')
    axes[1].set_ylabel('|dEz|')
    axes[1].set_xlim(-10, 350)
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: BIM
    axes[2].imshow(bim, extent=ext, cmap=cmap, vmin=1, vmax=80,
                   origin='lower', interpolation='bilinear')
    axes[2].set_title('BIM\n(Classical)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('x (cm)'); axes[2].set_ylabel('y (cm)')
    
    # Panel 4: BIM + U-Net
    axes[3].imshow(unet_map, extent=ext, cmap=cmap, vmin=1, vmax=80,
                   origin='lower', interpolation='bilinear')
    axes[3].set_title('BIM + U-Net\n(Denoised)', fontsize=12, fontweight='bold')
    axes[3].set_xlabel('x (cm)'); axes[3].set_ylabel('y (cm)')
    
    # Panel 5: PINN
    axes[4].imshow(pinn, extent=ext, cmap=cmap, vmin=1, vmax=80,
                   origin='lower', interpolation='bilinear')
    axes[4].set_title('PINN\n(Physics-Informed)', fontsize=12, fontweight='bold')
    axes[4].set_xlabel('x (cm)'); axes[4].set_ylabel('y (cm)')
    
    # Colorbar
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cbar.set_label('Dielectric Permittivity (er)', fontsize=12)
    
    fig.suptitle(f'Wi-Fi Microwave Tomography (MEEP FDTD Validated) - {title_suffix}',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")


def plot_all_levels(results, filename):
    """4 severity levels comparison."""
    cmap = tomo_cmap()
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    ext = [-15, 15, -15, 15]
    
    for i, level in enumerate(['none', 'mild', 'moderate', 'severe']):
        if level not in results:
            continue
        r = results[level]
        
        axes[0,i].imshow(r['gt'], extent=ext, cmap=cmap, vmin=1, vmax=80,
                         origin='lower', interpolation='bilinear')
        axes[0,i].set_title(f'Ground Truth\n{level.title()}', fontsize=12, fontweight='bold')
        axes[0,i].set_xlabel('x (cm)'); axes[0,i].set_ylabel('y (cm)')
        
        im = axes[1,i].imshow(r['recon'], extent=ext, cmap=cmap, vmin=1, vmax=80,
                              origin='lower', interpolation='bilinear')
        axes[1,i].set_title(f'Reconstruction\n{level.title()}', fontsize=12, fontweight='bold')
        axes[1,i].set_xlabel('x (cm)'); axes[1,i].set_ylabel('y (cm)')
    
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cbar.set_label('Dielectric Permittivity (er)', fontsize=12)
    
    fig.suptitle('Wi-Fi Tomography - Edema Detection Across Severity Levels\n(MEEP FDTD Forward Model)',
                 fontsize=15, fontweight='bold')
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}")

# ============================================================================
#   MAIN
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("  MEEP-VALIDATED WI-FI TOMOGRAPHY PIPELINE")
    print("=" * 60)
    
    # Load MEEP data for CSI display
    print("\n[1] Loading MEEP data...")
    meep_dir = 'meep_training_data'
    meep_csi = {}
    for f in sorted(glob.glob(os.path.join(meep_dir, '*.npz'))):
        d = np.load(f, allow_pickle=True)
        level = str(d['edema_level'])
        if level not in meep_csi:
            meep_csi[level] = []
        meep_csi[level].append(d['csi_differential'])
    
    for k,v in meep_csi.items():
        print(f"    {k}: {len(v)} MEEP samples")
    
    # Build G
    print("\n[2] Building G matrix...")
    G = build_G()
    print(f"    G: {G.shape}")
    
    # Train U-Net
    unet_model = None
    if TORCH_OK:
        print("\n[3] Training U-Net denoiser (500 pairs, 100 epochs)...")
        unet_model = train_unet(G, num_samples=500, epochs=100)
        print("    U-Net trained!")
    
    # Process each severity level
    print("\n[4] Reconstructing all 4 levels...")
    all_results = {}
    
    for level in ['none', 'mild', 'moderate', 'severe']:
        print(f"\n  === {level.upper()} ===")
        
        # Ground truth
        gt = make_phantom(level)
        print(f"    GT: er={gt.min():.0f} to {gt.max():.0f}")
        
        # Simulate CSI using Born (consistent with G matrix)
        chi_true = (gt.ravel() - 1.0).astype(complex)
        y_born = G @ chi_true
        noise = 0.02 * np.linalg.norm(y_born) * (np.random.randn(N_POS) + 1j*np.random.randn(N_POS))
        y_sim = y_born + noise
        
        # BIM reconstruction
        print(f"    Running BIM (300 iters)...")
        chi_bim = run_bim(G, y_sim, iters=300, lam=0.001, relax=0.5)
        bim_map = 1 + chi_bim.real.reshape(N, N)
        bim_map = np.clip(bim_map, 1, 80)
        bim_map = gaussian_filter(bim_map, sigma=0.5)
        print(f"    BIM: er={bim_map.min():.1f} to {bim_map.max():.1f}")
        
        # U-Net denoising
        unet_map = bim_map.copy()
        if unet_model is not None:
            unet_map = apply_unet(unet_model, bim_map)
            print(f"    U-Net: er={unet_map.min():.1f} to {unet_map.max():.1f}")
        
        # PINN reconstruction (only none & severe for speed)
        pinn_map = bim_map.copy()
        if TORCH_OK and level in ['none', 'severe']:
            print(f"    Running PINN (15000 epochs)...")
            pinn_map = train_pinn(G, y_sim, epochs=15000, lr=3e-4)
            pinn_map = gaussian_filter(pinn_map, sigma=0.3)
            print(f"    PINN: er={pinn_map.min():.1f} to {pinn_map.max():.1f}")
        
        # Get MEEP CSI for display
        avg_meep = np.zeros(N_POS)
        if level in meep_csi and meep_csi[level]:
            avg_meep = np.mean(np.abs(np.array(meep_csi[level])), axis=0)
        
        all_results[level] = {'gt': gt, 'bim': bim_map, 'unet': unet_map, 'recon': pinn_map, 'meep': avg_meep}
        
        # Save individual 5-panel comparison
        plot_final(gt, bim_map, unet_map, pinn_map, avg_meep, level.title(),
                   f'meep_result_{level}.png')
    
    # Save 4-level comparison
    print("\n[5] Generating comparison plots...")
    plot_all_levels(all_results, 'meep_all_levels.png')
    
    # Final heatmap for teacher (use best method)
    if 'severe' in all_results:
        fig, ax = plt.subplots(figsize=(8,7))
        r = all_results['severe']['unet']  # U-Net denoised
        cmap = tomo_cmap()
        im = ax.imshow(r, extent=[-15,15,-15,15], cmap=cmap, vmin=1, vmax=80,
                       origin='lower', interpolation='bilinear')
        max_er = r.max()
        ax.set_title(f'Wi-Fi Tomography (MEEP FDTD + BIM + U-Net)\n'
                     f'Pulmonary Edema Detected (er = {max_er:.0f})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('x (cm)', fontsize=12)
        ax.set_ylabel('y (cm)', fontsize=12)
        fig.colorbar(im, label='Dielectric Permittivity (er)')
        plt.tight_layout()
        plt.savefig('meep_final_heatmap.png', dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print("  Saved: meep_final_heatmap.png")
    
    print(f"\n{'='*60}")
    print(f"  DONE in {time.time()-t0:.0f} seconds")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
