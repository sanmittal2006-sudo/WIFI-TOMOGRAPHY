#!/usr/bin/env python3
"""
TWO-LUNG Wi-Fi Tomography Pipeline
====================================
Realistic chest phantom with:
  - Chest wall (outer ellipse, muscle εᵣ≈45)
  - Left lung (εᵣ=1, air-filled hollow cylinder)
  - Right lung (εᵣ=1, air-filled hollow cylinder)
  - Heart/mediastinum center (εᵣ≈60)
  - Edema = water in lung cylinder (εᵣ→60-78)

Full pipeline: Generate 2000 training pairs → Train U-Net → BIM+PINN+U-Net combined
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.linalg import lu_factor, lu_solve
import os, time, sys

sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")

N = 32; DOMAIN = 0.30; ANT_R = 0.18; N_POS = 16
FREQ = 2.4e9; C = 3e8; K0 = 2*np.pi*FREQ/C; DX = DOMAIN/N

UNET_TRAIN_SAMPLES = 2000
UNET_EPOCHS = 300
PINN_INIT_EPOCHS = 3000
PINN_PHYSICS_EPOCHS = 15000
BIM_ITERS = 400

def tomo_cmap():
    return LinearSegmentedColormap.from_list('tomo',
        ['#000033','#0000AA','#0066FF','#00CCFF','#00FFAA',
         '#66FF33','#CCFF00','#FFCC00','#FF6600','#FF0000','#990000'], N=256)

# ============================================================================
#   TWO-LUNG PHANTOM
# ============================================================================
def make_phantom_twolungs(edema='none', randomize=False):
    """
    3D-printed chest phantom (matches physical build):
      - Outer cylinder: chest wall liquid (water+glycerine, εᵣ=45)
      - Left lung:  hollow cylinder, AIR (εᵣ=1)
      - Right lung: hollow cylinder, AIR (εᵣ=1)
      - Heart: small cylinder at center (εᵣ=60)
      - Edema: water poured into right lung cylinder (εᵣ=60-78)
    """
    x = np.linspace(-DOMAIN/2+DX/2, DOMAIN/2-DX/2, N)
    xx, yy = np.meshgrid(x, x)

    eps = np.ones((N, N))  # air background

    # 1. Chest wall (outer ellipse) — muscle/fat
    chest_mask = ((xx/(0.13))**2 + (yy/(0.11))**2) <= 1.0
    eps[chest_mask] = 45.0

    # 2. Left lung (hollow cylinder = AIR, matches 3D print)
    left_lung = (((xx + 0.045)/0.05)**2 + ((yy)/0.07)**2) <= 1.0
    eps[left_lung] = 1.0   # AIR — empty cylinder

    # 3. Right lung (hollow cylinder = AIR, matches 3D print)
    right_lung = (((xx - 0.045)/0.05)**2 + ((yy)/0.07)**2) <= 1.0
    eps[right_lung] = 1.0  # AIR — empty cylinder

    # 4. Heart (center)
    heart = (xx**2 + yy**2) <= 0.02**2
    eps[heart] = 60.0

    # 5. Edema (fluid pocket in right lung)
    if edema == 'none':
        pass
    else:
        if randomize:
            # Random edema in either lung for training diversity
            side = np.random.choice([-1, 1])  # left or right
            cx = side * np.random.uniform(0.02, 0.07)
            cy = np.random.uniform(-0.04, 0.04)
            if edema == 'mild':
                rad = np.random.uniform(0.01, 0.02)
                er = np.random.uniform(40, 55)
            elif edema == 'moderate':
                rad = np.random.uniform(0.02, 0.035)
                er = np.random.uniform(55, 72)
            else:  # severe
                rad = np.random.uniform(0.03, 0.05)
                er = np.random.uniform(68, 78)
        else:
            # Fixed positions for display — edema in right lung
            params = {
                'mild':     {'cx': 0.05, 'cy': 0.02, 'rad': 0.015, 'er': 60},
                'moderate': {'cx': 0.045, 'cy': 0, 'rad': 0.025, 'er': 70},
                'severe':   {'cx': 0.045, 'cy': 0, 'rad': 0.04, 'er': 78},
            }
            p = params[edema]
            cx, cy, rad, er = p['cx'], p['cy'], p['rad'], p['er']

        edema_mask = np.sqrt((xx - cx)**2 + (yy - cy)**2) <= rad
        # Only place edema where there's already lung (air) tissue
        lung_region = (eps <= 1.0)
        eps[edema_mask & lung_region] = er

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
        angle = i * 2*np.pi / N_POS
        tx_pos = np.array([ANT_R*np.cos(angle), ANT_R*np.sin(angle)])
        for j in range(N*N):
            r1 = max(np.linalg.norm(pixels[j] - tx_pos), 1e-6)
            r2 = max(np.linalg.norm(rx_pos - pixels[j]), 1e-6)
            G[i,j] = (K0**2 * DX**2 / (4j)) * np.exp(1j*K0*(r1+r2)) / np.sqrt(r1*r2)
    return G

# ============================================================================
#   BIM (with LU factorization)
# ============================================================================
def run_bim_fast(G, y, lu_data, GH, iters=400):
    chi = np.zeros(N*N, dtype=complex)
    for i in range(iters):
        residual = y - G @ chi
        chi += 0.5 * lu_solve(lu_data, GH @ residual)
    return np.clip(1 + chi.real.reshape(N, N), 1, 80)

# ============================================================================
#   U-NET
# ============================================================================
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(True))
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1=ConvBlock(1,64); self.enc2=ConvBlock(64,128); self.enc3=ConvBlock(128,256)
        self.bottleneck=ConvBlock(256,512)
        self.up3=nn.ConvTranspose2d(512,256,2,stride=2); self.dec3=ConvBlock(512,256)
        self.up2=nn.ConvTranspose2d(256,128,2,stride=2); self.dec2=ConvBlock(256,128)
        self.up1=nn.ConvTranspose2d(128,64,2,stride=2); self.dec1=ConvBlock(128,64)
        self.final=nn.Conv2d(64,1,1); self.pool=nn.MaxPool2d(2)
    def forward(self, x):
        e1=self.enc1(x); e2=self.enc2(self.pool(e1)); e3=self.enc3(self.pool(e2))
        b=self.bottleneck(self.pool(e3))
        d3=self.dec3(torch.cat([self.up3(b),e3],1))
        d2=self.dec2(torch.cat([self.up2(d3),e2],1))
        d1=self.dec1(torch.cat([self.up1(d2),e1],1))
        return torch.sigmoid(self.final(d1))

# ============================================================================
#   PINN
# ============================================================================
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1))
    def forward(self, x):
        return 1.0 + 79.0 * torch.sigmoid(self.net(x))

def train_pinn_from_unet(G, y, unet_map):
    model = PINN()
    coords = torch.tensor([[(ix+.5)/N, (iy+.5)/N] for iy in range(N) for ix in range(N)], dtype=torch.float32)
    Gt = torch.tensor(G, dtype=torch.complex64)
    yt = torch.tensor(y, dtype=torch.complex64)
    unet_target = torch.tensor(unet_map.ravel(), dtype=torch.float32)

    # Phase 1: Learn U-Net output
    print(f"        Phase 1: Init PINN from U-Net ({PINN_INIT_EPOCHS} ep)...", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(PINN_INIT_EPOCHS):
        opt.zero_grad()
        loss = torch.mean((model(coords).squeeze() - unet_target)**2)
        loss.backward(); opt.step()
        if ep % 1000 == 0: print(f"          Init {ep}: MSE={loss.item():.4f}", flush=True)

    # Phase 2: Physics
    print(f"        Phase 2: Physics refinement ({PINN_PHYSICS_EPOCHS} ep)...", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PINN_PHYSICS_EPOCHS, eta_min=1e-6)
    for ep in range(PINN_PHYSICS_EPOCHS):
        opt.zero_grad()
        eps = model(coords).squeeze()
        yp = Gt @ (eps - 1).to(torch.complex64)
        phys = torch.mean(torch.abs(yp - yt)**2) / (torch.mean(torch.abs(yt)**2) + 1e-8)
        data = torch.mean((eps - unet_target)**2) / (torch.mean(unet_target**2) + 1e-8)
        e2 = eps.reshape(N, N)
        tv = torch.mean(torch.abs(e2[1:,:]-e2[:-1,:])) + torch.mean(torch.abs(e2[:,1:]-e2[:,:-1]))
        loss = phys + 0.3 * data + 0.01 * tv
        loss.backward(); opt.step(); sched.step()
        if ep % 5000 == 0:
            print(f"          Phys {ep}: L={loss.item():.4f} p={phys.item():.4f}", flush=True)

    with torch.no_grad():
        return model(coords).squeeze().numpy().reshape(N, N)

# ============================================================================
#   METRICS
# ============================================================================
def metrics(gt, r):
    g, r2 = gt/80, r/80; rmse = np.sqrt(np.mean((g-r2)**2))
    mg, mr = g.mean(), r2.mean(); sg, sr = g.std(), r2.std()
    cov = np.mean((g-mg)*(r2-mr)); c1, c2 = 1e-4, 9e-4
    ssim = ((2*mg*mr+c1)*(2*cov+c2)) / ((mg**2+mr**2+c1)*(sg**2+sr**2+c2))
    return rmse, ssim

def roi_permittivity(img, region='right_lung'):
    """Mean εᵣ in right lung region (where edema appears)"""
    x = np.linspace(-DOMAIN/2+DX/2, DOMAIN/2-DX/2, N)
    xx, yy = np.meshgrid(x, x)
    if region == 'right_lung':
        mask = (((xx - 0.045)/0.05)**2 + ((yy)/0.07)**2) <= 1.0
    else:
        mask = (((xx + 0.045)/0.05)**2 + ((yy)/0.07)**2) <= 1.0
    if mask.sum() == 0: return 0
    return img[mask].mean()

# ============================================================================
#   MAIN
# ============================================================================
def main():
    t0 = time.time()
    print("="*65)
    print("  TWO-LUNG Wi-Fi Tomography Pipeline")
    print("  Chest Model: Chest Wall + Left Lung + Right Lung + Heart")
    print("  Pipeline: BIM → U-Net → PINN (combined)")
    print("="*65, flush=True)

    # Show phantom layout
    print("\n  Phantom (matches 3D-printed physical build):")
    print("    Chest wall:  εᵣ = 45 (water+glycerine)")
    print("    Lungs:       εᵣ = 1  (AIR — hollow cylinders)")
    print("    Heart:       εᵣ = 60")
    print("    Edema:       εᵣ = 40-78 (water in lung cylinder)")

    G = build_G()
    GH = G.conj().T
    lu_data = lu_factor(GH @ G + 0.001 * np.eye(N*N))

    # ==================================================================
    #   STEP 1: Generate 2000 BIM/GT training pairs (two-lung phantom)
    # ==================================================================
    print(f"\n  STEP 1: Generating {UNET_TRAIN_SAMPLES} training pairs...", flush=True)
    X_train = []; Y_train = []
    levels = ['none', 'mild', 'moderate', 'severe']

    for i in range(UNET_TRAIN_SAMPLES):
        edema = np.random.choice(levels)
        gt = make_phantom_twolungs(edema, randomize=True)
        chi_true = (gt.ravel() - 1).astype(complex)
        y = G @ chi_true
        y += 0.02 * np.linalg.norm(y) * (np.random.randn(N_POS) + 1j*np.random.randn(N_POS))

        bim = gaussian_filter(run_bim_fast(G, y, lu_data, GH, iters=100), 0.5)
        X_train.append(bim / 80.0)
        Y_train.append(gt / 80.0)

        if (i+1) % 200 == 0:
            print(f"    {i+1}/{UNET_TRAIN_SAMPLES} pairs generated", flush=True)

    X_train = torch.tensor(np.array(X_train), dtype=torch.float32).unsqueeze(1)
    Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32).unsqueeze(1)
    print(f"    Training data: X={X_train.shape}, Y={Y_train.shape}", flush=True)

    # ==================================================================
    #   STEP 2: Train U-Net
    # ==================================================================
    print(f"\n  STEP 2: Training U-Net ({UNET_EPOCHS} epochs)...", flush=True)
    unet = UNet()
    opt = torch.optim.Adam(unet.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=UNET_EPOCHS, eta_min=1e-5)
    dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    best_loss = 1e9
    for epoch in range(UNET_EPOCHS):
        unet.train(); total_loss = 0
        for xb, yb in loader:
            opt.zero_grad()
            loss = nn.MSELoss()(unet(xb), yb)
            loss.backward(); opt.step()
            total_loss += loss.item()
        sched.step()
        avg = total_loss / len(loader)
        if avg < best_loss:
            best_loss = avg
            torch.save(unet.state_dict(), 'unet_twolungs.pth')
        if (epoch+1) % 30 == 0:
            print(f"    Epoch {epoch+1}/{UNET_EPOCHS}: loss={avg:.6f} (best={best_loss:.6f})", flush=True)

    unet.load_state_dict(torch.load('unet_twolungs.pth', weights_only=True))
    unet.eval()
    print(f"    U-Net trained! Best loss: {best_loss:.6f}", flush=True)

    # ==================================================================
    #   STEP 3: Run full pipeline on all severity levels
    # ==================================================================
    print(f"\n  STEP 3: Running BIM → U-Net → PINN on all levels...", flush=True)
    cmap = tomo_cmap(); ext = [-15, 15, -15, 15]
    results = {}

    for level in ['none', 'mild', 'moderate', 'severe']:
        print(f"\n    === {level.upper()} ===", flush=True)
        gt = make_phantom_twolungs(level, randomize=False)
        chi_t = (gt.ravel() - 1).astype(complex)
        y = G @ chi_t + 0.02*np.linalg.norm(G@chi_t)*(np.random.randn(N_POS)+1j*np.random.randn(N_POS))

        # BIM
        print(f"      BIM ({BIM_ITERS} iters)...", flush=True)
        bim = gaussian_filter(run_bim_fast(G, y, lu_data, GH, BIM_ITERS), 0.5)

        # U-Net
        print(f"      U-Net...", flush=True)
        with torch.no_grad():
            unet_map = np.clip(unet(torch.tensor(bim/80, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).squeeze().numpy()*80, 1, 80)

        # PINN (initialized from U-Net)
        print(f"      PINN (from U-Net)...", flush=True)
        pinn = np.clip(train_pinn_from_unet(G, y, unet_map), 1, 80)

        # Combined: 0.7*U-Net + 0.3*PINN
        combined = np.clip(gaussian_filter(0.7 * unet_map + 0.3 * pinn, 0.2), 1, 80)

        roi_gt = roi_permittivity(gt)
        roi_rec = roi_permittivity(combined)
        print(f"      GT right-lung εᵣ={roi_gt:.1f}  |  Reconstructed εᵣ={roi_rec:.1f}", flush=True)

        results[level] = {'gt':gt, 'bim':bim, 'unet':unet_map, 'pinn':pinn, 'combined':combined}
        np.savez(f'results_twolungs_{level}.npz', **results[level])

        # 5-panel plot
        rb,sb = metrics(gt,bim); ru,su = metrics(gt,unet_map); rc,sc = metrics(gt,combined)
        fig, axes = plt.subplots(1, 5, figsize=(30, 5.5))
        axes[0].imshow(gt, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].imshow(bim, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        axes[1].set_title(f'BIM\nSSIM={sb:.3f}', fontsize=11, fontweight='bold')
        axes[2].imshow(unet_map, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        axes[2].set_title(f'BIM+U-Net\nSSIM={su:.3f}', fontsize=11, fontweight='bold')
        axes[3].imshow(pinn, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        axes[3].set_title(f'PINN\nSSIM={metrics(gt,pinn)[1]:.3f}', fontsize=11, fontweight='bold')
        im = axes[4].imshow(combined, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        axes[4].set_title(f'BIM+U-Net+PINN\nSSIM={sc:.3f}', fontsize=11, fontweight='bold', color='#D32F2F')
        for ax in axes: ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02).set_label('Permittivity (εr)')
        fig.suptitle(f'Wi-Fi Tomography — {level.title()} Edema (Two-Lung Chest Model)\nPipeline: BIM → U-Net → PINN',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.savefig(f'final_{level}.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
        print(f"    Saved: final_{level}.png", flush=True)

    # ==================================================================
    #   4×4 COMPARISON
    # ==================================================================
    print("\n  Generating comparison plots...", flush=True)
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    for i, lv in enumerate(['none','mild','moderate','severe']):
        r = results[lv]
        axes[0,i].imshow(r['gt'], extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        axes[0,i].set_title(lv.title(), fontsize=13, fontweight='bold')
        axes[1,i].imshow(r['bim'], extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        _,s = metrics(r['gt'],r['bim']); axes[1,i].set_title(f'SSIM={s:.3f}', fontsize=11)
        axes[2,i].imshow(r['unet'], extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        _,s = metrics(r['gt'],r['unet']); axes[2,i].set_title(f'SSIM={s:.3f}', fontsize=11)
        im = axes[3,i].imshow(r['combined'], extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        _,s = metrics(r['gt'],r['combined']); axes[3,i].set_title(f'SSIM={s:.3f}', fontsize=11, color='#D32F2F', fontweight='bold')
    axes[0,0].set_ylabel('Ground Truth', fontsize=11, fontweight='bold')
    axes[1,0].set_ylabel('BIM', fontsize=11, fontweight='bold')
    axes[2,0].set_ylabel('BIM+U-Net', fontsize=11, fontweight='bold')
    axes[3,0].set_ylabel('BIM+U-Net+PINN', fontsize=11, fontweight='bold', color='#D32F2F')
    for ax in axes.flat: ax.set_xlabel('x (cm)', fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02).set_label('Permittivity (εr)')
    fig.suptitle('Wi-Fi Tomography: Two-Lung Chest Model\nEdema Detection Across Severity (BIM → U-Net → PINN)',
                fontsize=15, fontweight='bold')
    plt.savefig('final_all_levels.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
    print("  Saved: final_all_levels.png", flush=True)

    # ==================================================================
    #   DETECTION PLOT
    # ==================================================================
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    for i, lv in enumerate(['none','mild','moderate','severe']):
        im = axes[i].imshow(results[lv]['combined'], extent=ext, cmap=cmap, vmin=1, vmax=80,
                           origin='lower', interpolation='bilinear')
        roi = roi_permittivity(results[lv]['combined'])
        axes[i].set_title(f'{lv.title()}\nRight Lung εᵣ={roi:.1f}', fontsize=13, fontweight='bold')
        axes[i].set_xlabel('x (cm)'); axes[i].set_ylabel('y (cm)')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85).set_label('Permittivity (εr)')
    fig.suptitle('Pulmonary Edema Detection — Two-Lung Chest Model\n(Combined: BIM → U-Net → PINN)',
                fontsize=14, fontweight='bold')
    plt.savefig('final_detection.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
    print("  Saved: final_detection.png", flush=True)

    # ==================================================================
    #   SEVERITY BAR CHART
    # ==================================================================
    fig, ax = plt.subplots(figsize=(10, 6))
    levels_display = ['None', 'Mild', 'Moderate', 'Severe']
    gt_rois = [roi_permittivity(results[lv.lower()]['gt']) for lv in levels_display]
    recon_rois = [roi_permittivity(results[lv.lower()]['combined']) for lv in levels_display]
    x_pos = np.arange(4)
    bars1 = ax.bar(x_pos-0.2, gt_rois, 0.35, label='Ground Truth', color='#1565C0', edgecolor='black')
    bars2 = ax.bar(x_pos+0.2, recon_rois, 0.35, label='BIM+U-Net+PINN', color='#D32F2F', edgecolor='black')
    ax.set_xticks(x_pos); ax.set_xticklabels(levels_display, fontsize=12)
    ax.set_ylabel('Mean Permittivity (εᵣ) in Right Lung', fontsize=12)
    ax.set_title('Edema Severity Quantification — Two-Lung Model\n(Mean εᵣ in Right Lung Region)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')
    for b1, b2 in zip(bars1, bars2):
        ax.text(b1.get_x()+b1.get_width()/2, b1.get_height()+0.5, f'{b1.get_height():.1f}',
                ha='center', fontsize=10, fontweight='bold')
        ax.text(b2.get_x()+b2.get_width()/2, b2.get_height()+0.5, f'{b2.get_height():.1f}',
                ha='center', fontsize=10, fontweight='bold', color='#D32F2F')
    plt.savefig('final_severity_chart.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
    print("  Saved: final_severity_chart.png", flush=True)

    # ==================================================================
    #   METRICS TABLE
    # ==================================================================
    print(f"\n{'='*75}")
    print("  FINAL METRICS — TWO-LUNG CHEST MODEL")
    print(f"{'='*75}")
    print(f"  {'Level':<12} {'BIM':<8} {'U-Net':<8} {'Combined':<10} {'GT εᵣ':<8} {'Recon εᵣ':<8}")
    print(f"  {'-'*60}")
    for lv in ['none','mild','moderate','severe']:
        _,sb = metrics(results[lv]['gt'], results[lv]['bim'])
        _,su = metrics(results[lv]['gt'], results[lv]['unet'])
        _,sc = metrics(results[lv]['gt'], results[lv]['combined'])
        gr = roi_permittivity(results[lv]['gt'])
        cr = roi_permittivity(results[lv]['combined'])
        print(f"  {lv:<12} {sb:<8.4f} {su:<8.4f} {sc:<10.4f} {gr:<8.1f} {cr:<8.1f}")
    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*75}")
    print("  DONE!")

if __name__ == '__main__':
    main()
