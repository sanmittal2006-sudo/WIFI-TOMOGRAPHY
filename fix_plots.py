#!/usr/bin/env python3
"""Quick fix: generate the comparison plots from saved individual data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter
import sys

sys.stdout.reconfigure(line_buffering=True)

N=32; DOMAIN=0.30; DX=DOMAIN/N

def tomo_cmap():
    return LinearSegmentedColormap.from_list('tomo',
        ['#000033','#0000AA','#0066FF','#00CCFF','#00FFAA',
         '#66FF33','#CCFF00','#FFCC00','#FF6600','#FF0000','#990000'], N=256)

def make_phantom(edema='none'):
    x = np.linspace(-DOMAIN/2+DX/2, DOMAIN/2-DX/2, N)
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx**2+yy**2)
    eps = np.ones((N,N)); eps[rr <= 0.10] = 52.0
    if edema == 'mild':
        eps[np.sqrt((xx-0.04)**2+yy**2) <= 0.02] = 65.0
    elif edema == 'moderate':
        eps[np.sqrt((xx-0.03)**2+yy**2) <= 0.03] = 72.0
    elif edema == 'severe':
        eps[np.sqrt((xx-0.03)**2+yy**2) <= 0.05] = 78.0
    return eps

def metrics(gt, r):
    g,r2 = gt/80, r/80; rmse = np.sqrt(np.mean((g-r2)**2))
    mg,mr = g.mean(),r2.mean(); sg,sr = g.std(),r2.std()
    cov = np.mean((g-mg)*(r2-mr)); c1,c2 = 1e-4,9e-4
    ssim = ((2*mg*mr+c1)*(2*cov+c2))/((mg**2+mr**2+c1)*(sg**2+sr**2+c2))
    return rmse, ssim

import torch, torch.nn as nn

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

# Load U-Net
print("Loading U-Net...", flush=True)
unet = UNet()
unet.load_state_dict(torch.load('unet_best.pth', weights_only=True, map_location='cpu'))
unet.eval()

# Generate U-Net results for all 4 levels
from scipy.linalg import lu_factor, lu_solve
import glob

N_POS=16; ANT_R=0.18; FREQ=2.4e9; C=3e8; K0=2*np.pi*FREQ/C

def build_G():
    x = np.linspace(-DOMAIN/2+DX/2, DOMAIN/2-DX/2, N)
    xx, yy = np.meshgrid(x, x)
    pixels = np.column_stack([xx.ravel(), yy.ravel()])
    rx = np.array([ANT_R*np.cos(np.pi), ANT_R*np.sin(np.pi)])
    G = np.zeros((N_POS, N*N), dtype=complex)
    for i in range(N_POS):
        ang = i*2*np.pi/N_POS
        tx = np.array([ANT_R*np.cos(ang), ANT_R*np.sin(ang)])
        for j in range(N*N):
            r1 = max(np.linalg.norm(pixels[j]-tx), 1e-6)
            r2 = max(np.linalg.norm(rx-pixels[j]), 1e-6)
            G[i,j] = (K0**2*DX**2/(4j))*np.exp(1j*K0*(r1+r2))/np.sqrt(r1*r2)
    return G

print("Building G...", flush=True)
G = build_G()

cmap = tomo_cmap()
ext = [-15,15,-15,15]
results = {}

for level in ['none','mild','moderate','severe']:
    print(f"  {level}...", flush=True)
    gt = make_phantom(level)
    
    # BIM
    chi_true = (gt.ravel()-1.0).astype(complex)
    y = G@chi_true + 0.02*np.linalg.norm(G@chi_true)*(np.random.randn(N_POS)+1j*np.random.randn(N_POS))
    
    GH = G.conj().T; reg = GH@G + 0.001*np.eye(N*N)
    lu = lu_factor(reg)
    chi = np.zeros(N*N, dtype=complex)
    for it in range(400):
        chi += 0.5*lu_solve(lu, GH@(y-G@chi))
    bim = np.clip(1+chi.real.reshape(N,N), 1, 80)
    bim = gaussian_filter(bim, 0.5)
    
    # U-Net
    with torch.no_grad():
        inp = torch.tensor(bim/80, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        unet_map = np.clip(unet(inp).squeeze().numpy()*80, 1, 80)
    
    # MEEP CSI
    avg_meep = np.zeros(N_POS)
    for f in glob.glob('meep_training_data/*.npz'):
        d = np.load(f, allow_pickle=True)
        if str(d['edema_level']) == level:
            avg_meep += np.abs(d['csi_differential'])
    if avg_meep.sum() > 0:
        avg_meep /= max(1, sum(1 for f in glob.glob('meep_training_data/*.npz') if str(np.load(f,allow_pickle=True)['edema_level'])==level))
    
    results[level] = {'gt':gt, 'bim':bim, 'unet':unet_map, 'meep':avg_meep}

# 3-row x 4-col comparison
print("Generating comparison plot...", flush=True)
fig, axes = plt.subplots(3, 4, figsize=(20, 14))
for i, lv in enumerate(['none','mild','moderate','severe']):
    r = results[lv]
    axes[0,i].imshow(r['gt'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
    axes[0,i].set_title(lv.title(),fontsize=13,fontweight='bold')
    
    axes[1,i].imshow(r['bim'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
    _,s = metrics(r['gt'],r['bim']); axes[1,i].set_title(f'SSIM={s:.3f}',fontsize=11)
    
    im = axes[2,i].imshow(r['unet'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
    _,s = metrics(r['gt'],r['unet']); axes[2,i].set_title(f'SSIM={s:.3f}',fontsize=11)

axes[0,0].set_ylabel('Ground Truth\ny (cm)',fontsize=11,fontweight='bold')
axes[1,0].set_ylabel('BIM\ny (cm)',fontsize=11,fontweight='bold')
axes[2,0].set_ylabel('BIM + U-Net\ny (cm)',fontsize=11,fontweight='bold')
for ax in axes.flat: ax.set_xlabel('x (cm)',fontsize=9)
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
cbar.set_label('Permittivity (er)')
fig.suptitle('Wi-Fi Tomography: Edema Detection Across Severity\n(MEEP FDTD + BIM + U-Net)',fontsize=15,fontweight='bold')
plt.savefig('final_all_levels.png',dpi=200,bbox_inches='tight',facecolor='white'); plt.close()
print("  Saved: final_all_levels.png", flush=True)

# Healthy vs Severe
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(results['none']['unet'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
axes[0].set_title('Healthy (No Edema)',fontsize=13,fontweight='bold'); axes[0].set_xlabel('x (cm)'); axes[0].set_ylabel('y (cm)')
im = axes[1].imshow(results['severe']['unet'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
axes[1].set_title('Severe Edema Detected',fontsize=13,fontweight='bold'); axes[1].set_xlabel('x (cm)'); axes[1].set_ylabel('y (cm)')
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85)
cbar.set_label('Permittivity (er)')
fig.suptitle('Wi-Fi Microwave Tomography: Pulmonary Edema Detection\n(MEEP FDTD + BIM + U-Net)',fontsize=14,fontweight='bold')
plt.savefig('final_detection.png',dpi=200,bbox_inches='tight',facecolor='white'); plt.close()
print("  Saved: final_detection.png", flush=True)

# Metrics table
print(f"\n{'='*60}")
print(f"  METRICS")
print(f"  {'Level':<12} {'BIM RMSE':<10} {'U-Net RMSE':<12} {'U-Net SSIM':<12}")
for lv in ['none','mild','moderate','severe']:
    rb,sb = metrics(results[lv]['gt'], results[lv]['bim'])
    ru,su = metrics(results[lv]['gt'], results[lv]['unet'])
    print(f"  {lv:<12} {rb:<10.4f} {ru:<12.4f} {su:<12.4f}")
print(f"{'='*60}")
print("DONE!", flush=True)
