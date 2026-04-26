#!/usr/bin/env python3
"""
FIXED COMBINED PIPELINE — distinct severity levels
Key changes:
  1. More distinct phantoms (bigger size/permittivity gaps)
  2. Weighted fusion: 0.7*U-Net + 0.3*PINN (best of both)
  3. Quantitative severity metric (mean εᵣ in ROI)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.linalg import lu_factor, lu_solve
import glob, time, sys, torch, torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)

N=32; DOMAIN=0.30; ANT_R=0.18; N_POS=16
FREQ=2.4e9; C=3e8; K0=2*np.pi*FREQ/C; DX=DOMAIN/N

def tomo_cmap():
    return LinearSegmentedColormap.from_list('tomo',
        ['#000033','#0000AA','#0066FF','#00CCFF','#00FFAA',
         '#66FF33','#CCFF00','#FFCC00','#FF6600','#FF0000','#990000'], N=256)

def make_phantom(edema='none'):
    """
    FIXED: More distinct levels — clear visual difference
    None:     healthy lung (εᵣ=52), no fluid
    Mild:     small fluid pocket, 3cm diameter, εᵣ=60, offset 3cm
    Moderate: medium fluid, 5cm diameter, εᵣ=70, centered
    Severe:   large fluid, 8cm diameter, εᵣ=78, centered
    """
    x = np.linspace(-DOMAIN/2+DX/2, DOMAIN/2-DX/2, N)
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx**2+yy**2)
    eps = np.ones((N,N))
    eps[rr <= 0.10] = 52.0  # lung tissue

    if edema == 'mild':
        # Small pocket, off-center, lower permittivity
        eps[np.sqrt((xx-0.04)**2 + (yy-0.02)**2) <= 0.015] = 60.0
    elif edema == 'moderate':
        # Medium pocket, near-center, higher permittivity
        eps[np.sqrt((xx-0.01)**2 + (yy)**2) <= 0.035] = 70.0
    elif edema == 'severe':
        # Large fluid-filled region, centered, near-water permittivity
        eps[np.sqrt((xx)**2 + (yy)**2) <= 0.06] = 78.0
    return eps

def build_G():
    x = np.linspace(-DOMAIN/2+DX/2, DOMAIN/2-DX/2, N)
    xx, yy = np.meshgrid(x, x)
    pix = np.column_stack([xx.ravel(), yy.ravel()])
    rx = np.array([ANT_R*np.cos(np.pi), ANT_R*np.sin(np.pi)])
    G = np.zeros((N_POS, N*N), dtype=complex)
    for i in range(N_POS):
        a = i*2*np.pi/N_POS
        tx = np.array([ANT_R*np.cos(a), ANT_R*np.sin(a)])
        for j in range(N*N):
            r1 = max(np.linalg.norm(pix[j]-tx),1e-6)
            r2 = max(np.linalg.norm(rx-pix[j]),1e-6)
            G[i,j] = (K0**2*DX**2/(4j))*np.exp(1j*K0*(r1+r2))/np.sqrt(r1*r2)
    return G

# U-Net
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

# PINN
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1))
    def forward(self, x):
        return 1.0 + 79.0 * torch.sigmoid(self.net(x))

def train_pinn_from_unet(G, y, unet_map, epochs_init=3000, epochs_physics=15000):
    model = PINN()
    coords = torch.tensor([[(ix+.5)/N, (iy+.5)/N] for iy in range(N) for ix in range(N)], dtype=torch.float32)
    Gt = torch.tensor(G, dtype=torch.complex64)
    yt = torch.tensor(y, dtype=torch.complex64)
    unet_target = torch.tensor(unet_map.ravel(), dtype=torch.float32)

    # Phase 1: Learn U-Net output
    print(f"      Phase 1: Init from U-Net ({epochs_init} ep)...", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs_init):
        opt.zero_grad()
        loss = torch.mean((model(coords).squeeze() - unet_target)**2)
        loss.backward(); opt.step()
        if ep % 1000 == 0: print(f"        Init {ep}/{epochs_init}: MSE={loss.item():.4f}", flush=True)

    # Phase 2: Physics refinement (lighter TV to preserve detail)
    print(f"      Phase 2: Physics ({epochs_physics} ep)...", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_physics, eta_min=1e-6)
    for ep in range(epochs_physics):
        opt.zero_grad()
        eps = model(coords).squeeze()
        yp = Gt @ (eps - 1).to(torch.complex64)
        phys = torch.mean(torch.abs(yp - yt)**2) / (torch.mean(torch.abs(yt)**2) + 1e-8)
        data = torch.mean((eps - unet_target)**2) / (torch.mean(unet_target**2) + 1e-8)
        e2 = eps.reshape(N, N)
        tv = torch.mean(torch.abs(e2[1:,:]-e2[:-1,:])) + torch.mean(torch.abs(e2[:,1:]-e2[:,:-1]))
        # Keep data fidelity HIGH so PINN stays close to U-Net
        loss = phys + 0.3 * data + 0.01 * tv
        loss.backward(); opt.step(); sched.step()
        if ep % 5000 == 0:
            print(f"        Phys {ep}/{epochs_physics}: L={loss.item():.4f} p={phys.item():.4f} d={data.item():.4f}", flush=True)

    with torch.no_grad():
        return model(coords).squeeze().numpy().reshape(N, N)

def metrics(gt, r):
    g, r2 = gt/80, r/80; rmse = np.sqrt(np.mean((g-r2)**2))
    mg, mr = g.mean(), r2.mean(); sg, sr = g.std(), r2.std()
    cov = np.mean((g-mg)*(r2-mr)); c1, c2 = 1e-4, 9e-4
    ssim = ((2*mg*mr+c1)*(2*cov+c2)) / ((mg**2+mr**2+c1)*(sg**2+sr**2+c2))
    return rmse, ssim

def roi_permittivity(img):
    """Mean εᵣ in center 10×10 pixel region (the ROI)"""
    c = N//2; r = 5
    return img[c-r:c+r, c-r:c+r].mean()

def main():
    t0 = time.time()
    print("="*60)
    print("  FIXED COMBINED PIPELINE: BIM → U-Net → PINN")
    print("  Weighted fusion: 0.7*U-Net + 0.3*PINN")
    print("="*60, flush=True)

    meep_csi = {}
    for f in sorted(glob.glob('meep_training_data/*.npz')):
        d = np.load(f, allow_pickle=True); lv = str(d['edema_level'])
        meep_csi.setdefault(lv, []).append(d['csi_differential'])

    print("  Building G...", flush=True)
    G = build_G()
    GH = G.conj().T; lu = lu_factor(GH @ G + 0.001 * np.eye(N*N))

    print("  Loading U-Net...", flush=True)
    unet = UNet()
    unet.load_state_dict(torch.load('unet_best.pth', weights_only=True, map_location='cpu'))
    unet.eval()

    cmap = tomo_cmap(); ext = [-15,15,-15,15]
    results = {}

    for level in ['none', 'mild', 'moderate', 'severe']:
        print(f"\n  ===== {level.upper()} =====", flush=True)
        gt = make_phantom(level)
        chi_t = (gt.ravel() - 1).astype(complex)
        y = G @ chi_t + 0.02*np.linalg.norm(G@chi_t)*(np.random.randn(N_POS)+1j*np.random.randn(N_POS))

        # Step 1: BIM
        print("    BIM...", flush=True)
        chi = np.zeros(N*N, dtype=complex)
        for it in range(400):
            chi += 0.5 * lu_solve(lu, GH @ (y - G @ chi))
        bim = np.clip(1 + chi.real.reshape(N,N), 1, 80)
        bim = gaussian_filter(bim, 0.5)

        # Step 2: U-Net
        print("    U-Net...", flush=True)
        with torch.no_grad():
            unet_map = np.clip(unet(torch.tensor(bim/80, dtype=torch.float32).unsqueeze(0).unsqueeze(0)).squeeze().numpy()*80, 1, 80)

        # Step 3: PINN from U-Net
        print("    PINN (from U-Net)...", flush=True)
        pinn = np.clip(train_pinn_from_unet(G, y, unet_map, 3000, 15000), 1, 80)

        # Step 4: WEIGHTED FUSION — best of U-Net detail + PINN physics
        combined = np.clip(0.7 * unet_map + 0.3 * pinn, 1, 80)
        combined = gaussian_filter(combined, 0.2)  # very light smoothing

        print(f"    GT ROI εᵣ={roi_permittivity(gt):.1f}  |  Combined ROI εᵣ={roi_permittivity(combined):.1f}", flush=True)

        avg_meep = np.zeros(N_POS)
        if level in meep_csi:
            avg_meep = np.mean(np.abs(np.array(meep_csi[level])), axis=0)

        results[level] = {'gt':gt, 'bim':bim, 'unet':unet_map, 'pinn':pinn, 'combined':combined, 'meep':avg_meep}
        np.savez(f'results_final_{level}.npz', **results[level])

        # 5-panel plot
        rb,sb = metrics(gt,bim); ru,su = metrics(gt,unet_map); rc,sc = metrics(gt,combined)
        fig, axes = plt.subplots(1, 5, figsize=(28, 5))
        axes[0].imshow(gt, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        axes[0].set_title('Ground Truth', fontsize=12, fontweight='bold')
        axes[1].bar(np.arange(N_POS)*22.5, avg_meep, width=15, color='#2196F3', alpha=0.85, edgecolor='#1565C0')
        axes[1].set_title('MEEP FDTD CSI', fontsize=12, fontweight='bold'); axes[1].grid(True, alpha=0.3)
        axes[2].imshow(bim, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        axes[2].set_title(f'BIM\nSSIM={sb:.3f}', fontsize=11, fontweight='bold')
        axes[3].imshow(unet_map, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        axes[3].set_title(f'BIM+U-Net\nSSIM={su:.3f}', fontsize=11, fontweight='bold')
        im = axes[4].imshow(combined, extent=ext, cmap=cmap, vmin=1, vmax=80, origin='lower', interpolation='bilinear')
        axes[4].set_title(f'BIM+U-Net+PINN\nSSIM={sc:.3f}', fontsize=11, fontweight='bold', color='#D32F2F')
        for ax in axes: ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02).set_label('Permittivity (εr)')
        fig.suptitle(f'Wi-Fi Tomography — {level.title()} Edema\n(Combined: BIM → U-Net → PINN)',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.savefig(f'final_{level}.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
        print(f"  Saved: final_{level}.png", flush=True)

    # ============ 4×4 COMPARISON ============
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
    axes[0,0].set_ylabel('Ground Truth\ny (cm)', fontsize=11, fontweight='bold')
    axes[1,0].set_ylabel('BIM\ny (cm)', fontsize=11, fontweight='bold')
    axes[2,0].set_ylabel('BIM+U-Net\ny (cm)', fontsize=11, fontweight='bold')
    axes[3,0].set_ylabel('BIM+U-Net+PINN\ny (cm)', fontsize=11, fontweight='bold', color='#D32F2F')
    for ax in axes.flat: ax.set_xlabel('x (cm)', fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02).set_label('Permittivity (εr)')
    fig.suptitle('Wi-Fi Tomography: Combined Pipeline (BIM → U-Net → PINN)\nEdema Detection Across Severity',
                fontsize=15, fontweight='bold')
    plt.savefig('final_all_levels.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
    print("  Saved: final_all_levels.png", flush=True)

    # ============ DETECTION ============
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.5))
    for i, lv in enumerate(['none','mild','moderate','severe']):
        im = axes[i].imshow(results[lv]['combined'], extent=ext, cmap=cmap, vmin=1, vmax=80,
                           origin='lower', interpolation='bilinear')
        roi = roi_permittivity(results[lv]['combined'])
        axes[i].set_title(f'{lv.title()}\nROI εᵣ = {roi:.1f}', fontsize=13, fontweight='bold')
        axes[i].set_xlabel('x (cm)'); axes[i].set_ylabel('y (cm)')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85).set_label('Permittivity (εr)')
    fig.suptitle('Pulmonary Edema Severity Detection (BIM → U-Net → PINN)\nProgression: Healthy → Mild → Moderate → Severe',
                fontsize=14, fontweight='bold')
    plt.savefig('final_detection.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
    print("  Saved: final_detection.png", flush=True)

    # ============ SEVERITY BAR CHART ============
    fig, ax = plt.subplots(figsize=(10, 6))
    levels = ['None', 'Mild', 'Moderate', 'Severe']
    gt_rois = [roi_permittivity(results[lv.lower()]['gt']) for lv in levels]
    recon_rois = [roi_permittivity(results[lv.lower()]['combined']) for lv in levels]
    x = np.arange(4)
    bars1 = ax.bar(x-0.2, gt_rois, 0.35, label='Ground Truth', color='#1565C0', edgecolor='black')
    bars2 = ax.bar(x+0.2, recon_rois, 0.35, label='BIM+U-Net+PINN', color='#D32F2F', edgecolor='black')
    ax.set_xticks(x); ax.set_xticklabels(levels, fontsize=12)
    ax.set_ylabel('Mean Permittivity (εᵣ) in ROI', fontsize=12)
    ax.set_title('Edema Severity Quantification\n(Mean εᵣ in Central Region)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3, axis='y')
    for b1, b2 in zip(bars1, bars2):
        ax.text(b1.get_x()+b1.get_width()/2, b1.get_height()+0.5, f'{b1.get_height():.1f}',
                ha='center', fontsize=10, fontweight='bold')
        ax.text(b2.get_x()+b2.get_width()/2, b2.get_height()+0.5, f'{b2.get_height():.1f}',
                ha='center', fontsize=10, fontweight='bold', color='#D32F2F')
    plt.savefig('final_severity_chart.png', dpi=200, bbox_inches='tight', facecolor='white'); plt.close()
    print("  Saved: final_severity_chart.png", flush=True)

    # ============ METRICS ============
    print(f"\n{'='*70}")
    print("  FINAL METRICS — COMBINED PIPELINE")
    print(f"{'='*70}")
    print(f"  {'Level':<12} {'BIM SSIM':<10} {'U-Net SSIM':<12} {'Combined SSIM':<14} {'GT ROI εᵣ':<12} {'Recon ROI εᵣ':<12}")
    print(f"  {'-'*70}")
    for lv in ['none','mild','moderate','severe']:
        _,sb = metrics(results[lv]['gt'], results[lv]['bim'])
        _,su = metrics(results[lv]['gt'], results[lv]['unet'])
        _,sc = metrics(results[lv]['gt'], results[lv]['combined'])
        gr = roi_permittivity(results[lv]['gt'])
        cr = roi_permittivity(results[lv]['combined'])
        print(f"  {lv:<12} {sb:<10.4f} {su:<12.4f} {sc:<14.4f} {gr:<12.1f} {cr:<12.1f}")
    print(f"\n  Time: {(time.time()-t0)/60:.1f} min")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()
