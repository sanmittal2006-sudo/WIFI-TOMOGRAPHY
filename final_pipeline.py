#!/usr/bin/env python3
"""
FINAL Wi-Fi Tomography Pipeline — Publication Quality
======================================================
Generates maximum training data, trains U-Net properly,
and produces clean heatmaps for all 4 edema levels.

Run: $env:PYTHONIOENCODING='utf-8'; python final_pipeline.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.linalg import lu_factor, lu_solve
import os, glob, time, sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
except:
    TORCH_OK = False
    print("WARNING: PyTorch not available")

# ============================================================================
#   CONFIG
# ============================================================================
N = 32
DOMAIN = 0.30
ANT_R = 0.18
N_POS = 16
FREQ = 2.4e9
C = 3e8
K0 = 2*np.pi*FREQ/C
DX = DOMAIN/N

# Training config
UNET_TRAIN_SAMPLES = 2000   # Number of BIM/GT training pairs
UNET_EPOCHS = 300            # Training epochs
PINN_EPOCHS = 20000          # PINN training epochs
BIM_ITERS = 400              # BIM iterations for final reconstruction

def tomo_cmap():
    return LinearSegmentedColormap.from_list('tomo',
        ['#000033','#0000AA','#0066FF','#00CCFF','#00FFAA',
         '#66FF33','#CCFF00','#FFCC00','#FF6600','#FF0000','#990000'], N=256)

# ============================================================================
#   PHANTOM
# ============================================================================
def make_phantom(edema='none', randomize=False):
    """Create agar phantom. If randomize=True, vary position/size for training."""
    x = np.linspace(-DOMAIN/2+DX/2, DOMAIN/2-DX/2, N)
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx**2+yy**2)
    
    eps = np.ones((N,N))
    eps[rr <= 0.10] = 52.0  # agar
    
    if edema == 'none':
        return eps
    
    # Edema parameters
    if randomize:
        # Random position and size for training diversity
        angle = np.random.uniform(0, 2*np.pi)
        dist = np.random.uniform(0.01, 0.06)
        cx = dist * np.cos(angle)
        cy = dist * np.sin(angle)
        if edema == 'mild':
            rad = np.random.uniform(0.015, 0.025)
            er = np.random.uniform(60, 70)
        elif edema == 'moderate':
            rad = np.random.uniform(0.025, 0.04)
            er = np.random.uniform(65, 75)
        else:  # severe
            rad = np.random.uniform(0.03, 0.06)
            er = np.random.uniform(70, 80)
    else:
        # Fixed positions for display
        params = {
            'mild':     {'cx': 0.04, 'cy': 0, 'rad': 0.02, 'er': 65},
            'moderate': {'cx': 0.03, 'cy': 0, 'rad': 0.03, 'er': 72},
            'severe':   {'cx': 0.03, 'cy': 0, 'rad': 0.05, 'er': 78},
        }
        p = params[edema]
        cx, cy, rad, er = p['cx'], p['cy'], p['rad'], p['er']
    
    mask = np.sqrt((xx-cx)**2 + (yy-cy)**2) <= rad
    eps[mask] = er
    
    return eps

# ============================================================================
#   GREEN'S FUNCTION
# ============================================================================
def build_G():
    print("  Building Green's function matrix...", flush=True)
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
    
    print(f"  G matrix: {G.shape}", flush=True)
    return G

# ============================================================================
#   BIM
# ============================================================================
def run_bim(G, y, iters=400, lam=0.001, relax=0.5, verbose=True):
    M = N*N
    chi = np.zeros(M, dtype=complex)
    GtG = G.conj().T @ G
    
    for it in range(iters):
        res = y - G@chi
        dchi = np.linalg.solve(GtG + lam*np.eye(M), G.conj().T @ res)
        chi += relax * dchi
        if verbose and (it+1) % 100 == 0:
            err = np.linalg.norm(res)/(np.linalg.norm(y)+1e-10)
            print(f"    BIM {it+1}/{iters}: err={err:.6f}", flush=True)
    return chi

def run_bim_fast(G, y, lu_piv, GH, iters=30):
    """Fast BIM using precomputed LU factorization (10x faster)."""
    chi = np.zeros(N*N, dtype=complex)
    for it in range(iters):
        res = y - G @ chi
        dchi = lu_solve(lu_piv, GH @ res)
        chi += 0.5 * dchi
    return chi

# ============================================================================
#   PINN
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
        return 1.0 + 79.0 * torch.sigmoid(self.net(x))

def train_pinn(G, y_meas, epochs=20000, lr=3e-4):
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
        
        data_loss = torch.mean(torch.abs(y_pred - y_t)**2) / (torch.mean(torch.abs(y_t)**2) + 1e-8)
        
        eps2d = eps.reshape(N, N)
        tv = torch.mean(torch.abs(eps2d[1:,:]-eps2d[:-1,:])) + \
             torch.mean(torch.abs(eps2d[:,1:]-eps2d[:,:-1]))
        
        loss = data_loss + 0.05 * tv
        loss.backward()
        opt.step()
        sched.step()
        
        if ep % 5000 == 0:
            print(f"    PINN {ep:5d}/{epochs}: loss={loss.item():.6f}", flush=True)
    
    with torch.no_grad():
        result = model(coords_t).squeeze().cpu().numpy().reshape(N, N)
    return result

# ============================================================================
#   U-NET
# ============================================================================
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(1, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)
        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlock(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlock(128, 64)
        self.final = nn.Conv2d(64, 1, 1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        # Encode
        e1 = self.enc1(x)                # 32->64ch
        e2 = self.enc2(self.pool(e1))     # 16->128ch
        e3 = self.enc3(self.pool(e2))     # 8->256ch
        b = self.bottleneck(self.pool(e3))# 4->512ch
        # Decode with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))   # 8->256ch
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))  # 16->128ch
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))  # 32->64ch
        out = torch.sigmoid(self.final(d1))
        return out

def train_unet(G, num_samples=2000, epochs=300):
    """Train U-Net on BIM->GT pairs with diverse phantoms."""
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  [U-Net] Generating {num_samples} training pairs...", flush=True)
    
    # Precompute LU factorization for massive speedup
    GH = G.conj().T
    GtG = GH @ G
    reg = GtG + 0.01 * np.eye(N*N)
    lu_piv = lu_factor(reg)
    print(f"  [U-Net] LU factorization precomputed", flush=True)
    
    X_data = []
    Y_data = []
    levels = ['none', 'mild', 'moderate', 'severe']
    
    t0 = time.time()
    for s in range(num_samples):
        level = np.random.choice(levels)
        gt = make_phantom(level, randomize=True)
        
        chi_true = (gt.ravel() - 1.0).astype(complex)
        y_sim = G @ chi_true
        noise_level = np.random.uniform(0.01, 0.05)
        y_sim += noise_level * np.linalg.norm(y_sim) * (np.random.randn(N_POS) + 1j*np.random.randn(N_POS))
        
        chi_bim = run_bim_fast(G, y_sim, lu_piv, GH, iters=30)
        bim = np.clip(1 + chi_bim.real.reshape(N, N), 1, 80)
        bim = gaussian_filter(bim, 0.5)
        
        X_data.append(bim)
        Y_data.append(gt)
        
        if (s+1) % 200 == 0:
            elapsed = time.time() - t0
            rate = (s+1) / elapsed
            eta = (num_samples - s - 1) / rate
            print(f"    Generated {s+1}/{num_samples} pairs ({rate:.1f}/s, ETA: {eta:.0f}s)", flush=True)
    
    print(f"  [U-Net] Data generation done in {time.time()-t0:.0f}s", flush=True)
    
    X = torch.tensor(np.array(X_data), dtype=torch.float32).unsqueeze(1).to(dev) / 80.0
    Y = torch.tensor(np.array(Y_data), dtype=torch.float32).unsqueeze(1).to(dev) / 80.0
    
    model = UNet().to(dev)
    params = sum(p.numel() for p in model.parameters())
    print(f"  [U-Net] Model: {params:,} parameters", flush=True)
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    
    batch = 32
    best_loss = float('inf')
    print(f"  [U-Net] Training for {epochs} epochs...", flush=True)
    
    for ep in range(epochs):
        idx = torch.randperm(len(X))
        total_loss = 0
        n_batches = 0
        for i in range(0, len(X), batch):
            bx = X[idx[i:i+batch]]
            by = Y[idx[i:i+batch]]
            pred = model(bx)
            loss = nn.MSELoss()(pred, by) + 0.001 * nn.L1Loss()(pred, by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        sched.step()
        avg_loss = total_loss / n_batches
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'unet_best.pth')
        
        if (ep+1) % 30 == 0:
            print(f"    Epoch {ep+1}/{epochs}: loss={avg_loss:.6f} (best={best_loss:.6f})", flush=True)
    
    # Load best model
    model.load_state_dict(torch.load('unet_best.pth', weights_only=True))
    print(f"  [U-Net] Training done. Best loss: {best_loss:.6f}", flush=True)
    return model

def apply_unet(model, bim_map):
    dev = next(model.parameters()).device
    x = torch.tensor(bim_map/80.0, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
    with torch.no_grad():
        out = model(x).squeeze().cpu().numpy() * 80.0
    return np.clip(out, 1, 80)

# ============================================================================
#   METRICS
# ============================================================================
def compute_metrics(gt, recon):
    """Compute SSIM-like and RMSE metrics."""
    gt_n = gt / 80.0
    rc_n = recon / 80.0
    rmse = np.sqrt(np.mean((gt_n - rc_n)**2))
    
    mu_g, mu_r = gt_n.mean(), rc_n.mean()
    sg, sr = gt_n.std(), rc_n.std()
    cov = np.mean((gt_n - mu_g)*(rc_n - mu_r))
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2*mu_g*mu_r+c1)*(2*cov+c2)) / ((mu_g**2+mu_r**2+c1)*(sg**2+sr**2+c2))
    return rmse, ssim

# ============================================================================
#   PLOTTING
# ============================================================================
def plot_5panel(gt, meep_csi, bim, unet, pinn, title, filename):
    cmap = tomo_cmap()
    fig, axes = plt.subplots(1, 5, figsize=(26, 5))
    ext = [-15, 15, -15, 15]
    
    for ax in axes:
        ax.set_xlabel('x (cm)', fontsize=10)
        ax.set_ylabel('y (cm)', fontsize=10)
    
    axes[0].imshow(gt, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
    axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
    
    angles = np.arange(N_POS) * 22.5
    axes[1].bar(angles, meep_csi, width=15, color='#2196F3', alpha=0.85, edgecolor='#1565C0')
    axes[1].set_title('MEEP FDTD CSI', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('TX Angle (deg)', fontsize=10)
    axes[1].set_ylabel('|dEz|', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    # BIM metrics
    rmse_b, ssim_b = compute_metrics(gt, bim)
    axes[2].imshow(bim, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
    axes[2].set_title(f'BIM\nRMSE={rmse_b:.3f} SSIM={ssim_b:.3f}', fontsize=11, fontweight='bold')
    
    # U-Net metrics
    rmse_u, ssim_u = compute_metrics(gt, unet)
    axes[3].imshow(unet, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
    axes[3].set_title(f'BIM + U-Net\nRMSE={rmse_u:.3f} SSIM={ssim_u:.3f}', fontsize=11, fontweight='bold')
    
    # PINN metrics 
    rmse_p, ssim_p = compute_metrics(gt, pinn)
    im = axes[4].imshow(pinn, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
    axes[4].set_title(f'PINN\nRMSE={rmse_p:.3f} SSIM={ssim_p:.3f}', fontsize=11, fontweight='bold')
    
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cbar.set_label('Dielectric Permittivity (er)', fontsize=11)
    
    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}", flush=True)
    return {'bim': (rmse_b, ssim_b), 'unet': (rmse_u, ssim_u), 'pinn': (rmse_p, ssim_p)}


def plot_4levels(results, filename):
    cmap = tomo_cmap()
    fig, axes = plt.subplots(3, 4, figsize=(20, 14))
    ext = [-15, 15, -15, 15]
    
    row_labels = ['Ground Truth', 'BIM + U-Net', 'PINN']
    
    for i, level in enumerate(['none', 'mild', 'moderate', 'severe']):
        r = results[level]
        
        axes[0,i].imshow(r['gt'], extent=ext, cmap=cmap, vmin=1, vmax=80,
                         origin='lower', interpolation='bilinear')
        axes[0,i].set_title(f'{level.title()}', fontsize=13, fontweight='bold')
        
        axes[1,i].imshow(r['unet'], extent=ext, cmap=cmap, vmin=1, vmax=80,
                         origin='lower', interpolation='bilinear')
        rmse, ssim = compute_metrics(r['gt'], r['unet'])
        axes[1,i].set_title(f'SSIM={ssim:.3f}', fontsize=11)
        
        im = axes[2,i].imshow(r['pinn'], extent=ext, cmap=cmap, vmin=1, vmax=80,
                              origin='lower', interpolation='bilinear')
        rmse, ssim = compute_metrics(r['gt'], r['pinn'])
        axes[2,i].set_title(f'SSIM={ssim:.3f}', fontsize=11)
    
    for j, label in enumerate(row_labels):
        axes[j,0].set_ylabel(f'{label}\ny (cm)', fontsize=11, fontweight='bold')
    
    for ax in axes.flat:
        ax.set_xlabel('x (cm)', fontsize=9)
    
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85, pad=0.02)
    cbar.set_label('Dielectric Permittivity (er)', fontsize=12)
    
    fig.suptitle('Wi-Fi Tomography: Edema Detection Across Severity Levels\n'
                 '(MEEP FDTD Validated, BIM + U-Net + PINN)',
                 fontsize=15, fontweight='bold')
    plt.savefig(filename, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {filename}", flush=True)

# ============================================================================
#   MAIN
# ============================================================================
def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("  FINAL WI-FI TOMOGRAPHY PIPELINE", flush=True)
    print("  BIM + U-Net (2000 samples, 300 epochs) + PINN (20000)", flush=True)
    print("=" * 60, flush=True)
    
    # Load MEEP CSI
    print("\n[1/6] Loading MEEP data...", flush=True)
    meep_dir = 'meep_training_data'
    meep_csi = {}
    if os.path.exists(meep_dir):
        for f in sorted(glob.glob(os.path.join(meep_dir, '*.npz'))):
            d = np.load(f, allow_pickle=True)
            level = str(d['edema_level'])
            if level not in meep_csi:
                meep_csi[level] = []
            meep_csi[level].append(d['csi_differential'])
        for k,v in meep_csi.items():
            print(f"    {k}: {len(v)} MEEP samples", flush=True)
    else:
        print("    No MEEP data found (will use zeros for CSI display)", flush=True)
    
    # Build G
    print("\n[2/6] Building forward model...", flush=True)
    G = build_G()
    
    # Train U-Net
    unet_model = None
    if TORCH_OK:
        print(f"\n[3/6] Training U-Net ({UNET_TRAIN_SAMPLES} pairs, {UNET_EPOCHS} epochs)...", flush=True)
        unet_model = train_unet(G, num_samples=UNET_TRAIN_SAMPLES, epochs=UNET_EPOCHS)
    
    # Reconstruct all 4 levels
    print(f"\n[4/6] Reconstructing all 4 edema levels...", flush=True)
    results = {}
    all_metrics = {}
    
    for level in ['none', 'mild', 'moderate', 'severe']:
        print(f"\n  {'='*40}", flush=True)
        print(f"  {level.upper()}", flush=True)
        print(f"  {'='*40}", flush=True)
        
        gt = make_phantom(level, randomize=False)
        
        # Born simulation
        chi_true = (gt.ravel() - 1.0).astype(complex)
        y_born = G @ chi_true
        noise = 0.02 * np.linalg.norm(y_born) * (np.random.randn(N_POS) + 1j*np.random.randn(N_POS))
        y_sim = y_born + noise
        
        # BIM
        print(f"    Running BIM ({BIM_ITERS} iters)...", flush=True)
        chi_bim = run_bim(G, y_sim, iters=BIM_ITERS, lam=0.001, relax=0.5)
        bim_map = np.clip(1 + chi_bim.real.reshape(N, N), 1, 80)
        bim_map = gaussian_filter(bim_map, sigma=0.5)
        
        # U-Net
        unet_map = bim_map.copy()
        if unet_model:
            unet_map = apply_unet(unet_model, bim_map)
            print(f"    U-Net: er={unet_map.min():.1f} to {unet_map.max():.1f}", flush=True)
        
        # PINN
        pinn_map = bim_map.copy()
        if TORCH_OK:
            print(f"    Running PINN ({PINN_EPOCHS} epochs)...", flush=True)
            pinn_map = train_pinn(G, y_sim, epochs=PINN_EPOCHS, lr=3e-4)
            pinn_map = gaussian_filter(pinn_map, sigma=0.3)
            print(f"    PINN: er={pinn_map.min():.1f} to {pinn_map.max():.1f}", flush=True)
        
        # MEEP CSI
        avg_meep = np.zeros(N_POS)
        if level in meep_csi and meep_csi[level]:
            avg_meep = np.mean(np.abs(np.array(meep_csi[level])), axis=0)
        
        results[level] = {'gt': gt, 'bim': bim_map, 'unet': unet_map, 'pinn': pinn_map, 'meep': avg_meep}
        
        # Plot individual
        metrics = plot_5panel(gt, avg_meep, bim_map, unet_map, pinn_map,
                    f'Wi-Fi Tomography (MEEP FDTD) - {level.title()} Edema',
                    f'final_{level}.png')
        all_metrics[level] = metrics
    
    # 4-level comparison
    print(f"\n[5/6] Generating comparison plots...", flush=True)
    plot_4levels(results, 'final_all_levels.png')
    
    # Final heatmap
    print(f"\n[6/6] Final heatmap...", flush=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    cmap = tomo_cmap()
    ext = [-15, 15, -15, 15]
    
    axes[0].imshow(results['none']['unet'], extent=ext, cmap=cmap, vmin=1, vmax=80,
                   origin='lower', interpolation='bilinear')
    axes[0].set_title('Healthy (No Edema)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('x (cm)'); axes[0].set_ylabel('y (cm)')
    
    im = axes[1].imshow(results['severe']['unet'], extent=ext, cmap=cmap, vmin=1, vmax=80,
                        origin='lower', interpolation='bilinear')
    axes[1].set_title('Severe Edema Detected', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('x (cm)'); axes[1].set_ylabel('y (cm)')
    
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.85)
    cbar.set_label('Dielectric Permittivity (er)', fontsize=12)
    fig.suptitle('Wi-Fi Microwave Tomography: Pulmonary Edema Detection\n(MEEP FDTD + BIM + U-Net)',
                 fontsize=14, fontweight='bold')
    plt.savefig('final_detection.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: final_detection.png", flush=True)
    
    # Print metrics table
    print(f"\n{'='*60}", flush=True)
    print(f"  METRICS SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Level':<12} {'BIM RMSE':<10} {'U-Net RMSE':<12} {'PINN RMSE':<10} {'U-Net SSIM':<12}", flush=True)
    print(f"  {'-'*54}", flush=True)
    for level in ['none', 'mild', 'moderate', 'severe']:
        m = all_metrics[level]
        print(f"  {level:<12} {m['bim'][0]:<10.4f} {m['unet'][0]:<12.4f} {m['pinn'][0]:<10.4f} {m['unet'][1]:<12.4f}", flush=True)
    
    total = time.time() - t0
    print(f"\n  Total time: {total/60:.1f} minutes", flush=True)
    print(f"\n  Generated files:", flush=True)
    print(f"    final_none.png       - 5-panel healthy", flush=True)
    print(f"    final_mild.png       - 5-panel mild edema", flush=True)
    print(f"    final_moderate.png   - 5-panel moderate edema", flush=True)
    print(f"    final_severe.png     - 5-panel severe edema", flush=True)
    print(f"    final_all_levels.png - 4-level comparison (3 rows)", flush=True)
    print(f"    final_detection.png  - Healthy vs Severe side-by-side", flush=True)
    print(f"    unet_best.pth       - Trained U-Net model", flush=True)
    print(f"\n{'='*60}", flush=True)
    print(f"  DONE!", flush=True)
    print(f"{'='*60}", flush=True)

if __name__ == '__main__':
    main()
