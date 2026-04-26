#!/usr/bin/env python3
"""
PINN-CENTERED Final Pipeline — PINN is the star
Generates all plots with PINN prominently featured.
Saves intermediate .npy so we never lose data again.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.linalg import lu_factor, lu_solve
import os, glob, time, sys, torch, torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)

N=32; DOMAIN=0.30; ANT_R=0.18; N_POS=16
FREQ=2.4e9; C=3e8; K0=2*np.pi*FREQ/C; DX=DOMAIN/N

def tomo_cmap():
    return LinearSegmentedColormap.from_list('tomo',
        ['#000033','#0000AA','#0066FF','#00CCFF','#00FFAA',
         '#66FF33','#CCFF00','#FFCC00','#FF6600','#FF0000','#990000'], N=256)

def make_phantom(edema='none'):
    x = np.linspace(-DOMAIN/2+DX/2, DOMAIN/2-DX/2, N)
    xx, yy = np.meshgrid(x, x)
    rr = np.sqrt(xx**2+yy**2)
    eps = np.ones((N,N)); eps[rr<=0.10]=52.0
    if edema=='mild': eps[np.sqrt((xx-0.04)**2+yy**2)<=0.02]=65.0
    elif edema=='moderate': eps[np.sqrt((xx-0.03)**2+yy**2)<=0.03]=72.0
    elif edema=='severe': eps[np.sqrt((xx-0.03)**2+yy**2)<=0.05]=78.0
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

def run_bim(G, y, iters=400):
    GH=G.conj().T; lu=lu_factor(GH@G+0.001*np.eye(N*N))
    chi=np.zeros(N*N, dtype=complex)
    for it in range(iters):
        chi += 0.5*lu_solve(lu, GH@(y-G@chi))
        if (it+1)%100==0: print(f"    BIM {it+1}/{iters}", flush=True)
    return chi

# U-Net (must match exact class/attribute names from training)
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
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(2,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),
            nn.Linear(256,256),nn.GELU(),nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))
    def forward(s,x): return 1.0+79.0*torch.sigmoid(s.net(x))

def train_pinn(G, y, epochs=20000):
    model=PINN(); opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-6)
    coords=torch.tensor([[(ix+.5)/N,(iy+.5)/N] for iy in range(N) for ix in range(N)],dtype=torch.float32)
    Gt=torch.tensor(G,dtype=torch.complex64); yt=torch.tensor(y,dtype=torch.complex64)
    for ep in range(epochs):
        opt.zero_grad()
        eps=model(coords).squeeze(); yp=Gt@(eps-1).to(torch.complex64)
        dl=torch.mean(torch.abs(yp-yt)**2)/(torch.mean(torch.abs(yt)**2)+1e-8)
        e2=eps.reshape(N,N)
        tv=torch.mean(torch.abs(e2[1:,:]-e2[:-1,:]))+torch.mean(torch.abs(e2[:,1:]-e2[:,:-1]))
        loss=dl+0.05*tv; loss.backward(); opt.step(); sched.step()
        if ep%5000==0: print(f"    PINN {ep}/{epochs}: loss={loss.item():.6f}", flush=True)
    with torch.no_grad(): return model(coords).squeeze().numpy().reshape(N,N)

def metrics(gt, r):
    g,r2=gt/80,r/80; rmse=np.sqrt(np.mean((g-r2)**2))
    mg,mr=g.mean(),r2.mean(); sg,sr=g.std(),r2.std()
    cov=np.mean((g-mg)*(r2-mr)); c1,c2=1e-4,9e-4
    ssim=((2*mg*mr+c1)*(2*cov+c2))/((mg**2+mr**2+c1)*(sg**2+sr**2+c2))
    return rmse, ssim

def main():
    t0=time.time()
    print("="*60); print("  PINN-CENTERED FINAL PIPELINE"); print("="*60, flush=True)

    # Load MEEP CSI
    meep_csi={}
    for f in sorted(glob.glob('meep_training_data/*.npz')):
        d=np.load(f,allow_pickle=True); lv=str(d['edema_level'])
        meep_csi.setdefault(lv,[]).append(d['csi_differential'])

    print("  Building G...", flush=True)
    G=build_G()

    # Load U-Net
    print("  Loading U-Net...", flush=True)
    unet=UNet(); unet.load_state_dict(torch.load('unet_best.pth',weights_only=True,map_location='cpu')); unet.eval()

    cmap=tomo_cmap(); ext=[-15,15,-15,15]
    results={}

    for level in ['none','mild','moderate','severe']:
        print(f"\n  ===== {level.upper()} =====", flush=True)
        gt=make_phantom(level)
        chi_t=(gt.ravel()-1).astype(complex)
        y=G@chi_t+0.02*np.linalg.norm(G@chi_t)*(np.random.randn(N_POS)+1j*np.random.randn(N_POS))

        # BIM
        bim=np.clip(1+run_bim(G,y,400).real.reshape(N,N),1,80)
        bim=gaussian_filter(bim,0.5)

        # U-Net
        with torch.no_grad():
            unet_map=np.clip(unet(torch.tensor(bim/80,dtype=torch.float32).unsqueeze(0).unsqueeze(0)).squeeze().numpy()*80,1,80)

        # PINN (THE STAR)
        print(f"    >>> PINN (20000 epochs) <<<", flush=True)
        pinn=np.clip(gaussian_filter(train_pinn(G,y,20000),0.3),1,80)

        # MEEP CSI
        avg_meep=np.zeros(N_POS)
        if level in meep_csi: avg_meep=np.mean(np.abs(np.array(meep_csi[level])),axis=0)

        results[level]={'gt':gt,'bim':bim,'unet':unet_map,'pinn':pinn,'meep':avg_meep}

        # Save numpy arrays so we NEVER lose this data
        np.savez(f'results_{level}.npz', gt=gt, bim=bim, unet=unet_map, pinn=pinn, meep=avg_meep)
        print(f"    Saved: results_{level}.npz", flush=True)

        # Individual 5-panel
        rb,sb=metrics(gt,bim); ru,su=metrics(gt,unet_map); rp,sp=metrics(gt,pinn)
        fig,axes=plt.subplots(1,5,figsize=(26,5))
        axes[0].imshow(gt,extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
        axes[0].set_title('Ground Truth',fontsize=12,fontweight='bold')
        axes[1].bar(np.arange(N_POS)*22.5,avg_meep,width=15,color='#2196F3',alpha=0.85,edgecolor='#1565C0')
        axes[1].set_title('MEEP FDTD CSI',fontsize=12,fontweight='bold'); axes[1].grid(True,alpha=0.3)
        axes[2].imshow(bim,extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
        axes[2].set_title(f'BIM\nRMSE={rb:.3f} SSIM={sb:.3f}',fontsize=11,fontweight='bold')
        axes[3].imshow(unet_map,extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
        axes[3].set_title(f'BIM+U-Net\nRMSE={ru:.3f} SSIM={su:.3f}',fontsize=11,fontweight='bold')
        im=axes[4].imshow(pinn,extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
        axes[4].set_title(f'PINN\nRMSE={rp:.3f} SSIM={sp:.3f}',fontsize=11,fontweight='bold')
        for ax in axes: ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
        fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.85,pad=0.02).set_label('Permittivity (er)')
        fig.suptitle(f'Wi-Fi Tomography (MEEP FDTD + PINN) - {level.title()} Edema',fontsize=14,fontweight='bold',y=1.02)
        plt.savefig(f'final_{level}.png',dpi=200,bbox_inches='tight',facecolor='white'); plt.close()
        print(f"  Saved: final_{level}.png", flush=True)

    # ============ 4-LEVEL COMPARISON WITH PINN ============
    print("\n  Generating comparison plots WITH PINN...", flush=True)
    fig,axes=plt.subplots(4,4,figsize=(20,18))
    for i,lv in enumerate(['none','mild','moderate','severe']):
        r=results[lv]
        axes[0,i].imshow(r['gt'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
        axes[0,i].set_title(lv.title(),fontsize=13,fontweight='bold')
        axes[1,i].imshow(r['bim'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
        _,s=metrics(r['gt'],r['bim']); axes[1,i].set_title(f'SSIM={s:.3f}',fontsize=11)
        axes[2,i].imshow(r['unet'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
        _,s=metrics(r['gt'],r['unet']); axes[2,i].set_title(f'SSIM={s:.3f}',fontsize=11)
        im=axes[3,i].imshow(r['pinn'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
        _,s=metrics(r['gt'],r['pinn']); axes[3,i].set_title(f'SSIM={s:.3f}',fontsize=11)
    axes[0,0].set_ylabel('Ground Truth\ny (cm)',fontsize=11,fontweight='bold')
    axes[1,0].set_ylabel('BIM\ny (cm)',fontsize=11,fontweight='bold')
    axes[2,0].set_ylabel('BIM + U-Net\ny (cm)',fontsize=11,fontweight='bold')
    axes[3,0].set_ylabel('PINN\ny (cm)',fontsize=11,fontweight='bold')
    for ax in axes.flat: ax.set_xlabel('x (cm)',fontsize=9)
    cbar=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.85,pad=0.02)
    cbar.set_label('Permittivity (er)')
    fig.suptitle('Wi-Fi Tomography: Edema Detection — All Methods Comparison\n(MEEP FDTD + BIM + U-Net + PINN)',fontsize=15,fontweight='bold')
    plt.savefig('final_all_levels.png',dpi=200,bbox_inches='tight',facecolor='white'); plt.close()
    print("  Saved: final_all_levels.png", flush=True)

    # ============ PINN DETECTION (Healthy vs Severe) ============
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    axes[0].imshow(results['none']['pinn'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
    axes[0].set_title('Healthy (No Edema)',fontsize=13,fontweight='bold');axes[0].set_xlabel('x (cm)');axes[0].set_ylabel('y (cm)')
    im=axes[1].imshow(results['severe']['pinn'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
    axes[1].set_title('Severe Edema Detected',fontsize=13,fontweight='bold');axes[1].set_xlabel('x (cm)');axes[1].set_ylabel('y (cm)')
    cbar=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.85)
    cbar.set_label('Permittivity (er)')
    fig.suptitle('Wi-Fi Microwave Tomography: Pulmonary Edema Detection\n(Physics-Informed Neural Network — PINN)',fontsize=14,fontweight='bold')
    plt.savefig('final_detection_pinn.png',dpi=200,bbox_inches='tight',facecolor='white'); plt.close()
    print("  Saved: final_detection_pinn.png", flush=True)

    # ============ U-Net DETECTION ============
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    axes[0].imshow(results['none']['unet'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
    axes[0].set_title('Healthy (No Edema)',fontsize=13,fontweight='bold');axes[0].set_xlabel('x (cm)');axes[0].set_ylabel('y (cm)')
    im=axes[1].imshow(results['severe']['unet'],extent=ext,cmap=cmap,vmin=1,vmax=80,origin='lower',interpolation='bilinear')
    axes[1].set_title('Severe Edema Detected',fontsize=13,fontweight='bold');axes[1].set_xlabel('x (cm)');axes[1].set_ylabel('y (cm)')
    cbar=fig.colorbar(im,ax=axes.ravel().tolist(),shrink=0.85)
    cbar.set_label('Permittivity (er)')
    fig.suptitle('Wi-Fi Microwave Tomography: Pulmonary Edema Detection\n(BIM + U-Net Denoiser)',fontsize=14,fontweight='bold')
    plt.savefig('final_detection_unet.png',dpi=200,bbox_inches='tight',facecolor='white'); plt.close()
    print("  Saved: final_detection_unet.png", flush=True)

    # Metrics
    print(f"\n{'='*60}"); print("  METRICS SUMMARY"); print(f"{'='*60}")
    print(f"  {'Level':<12} {'BIM SSIM':<10} {'U-Net SSIM':<12} {'PINN SSIM':<10}")
    print(f"  {'-'*44}")
    for lv in ['none','mild','moderate','severe']:
        _,sb=metrics(results[lv]['gt'],results[lv]['bim'])
        _,su=metrics(results[lv]['gt'],results[lv]['unet'])
        _,sp=metrics(results[lv]['gt'],results[lv]['pinn'])
        print(f"  {lv:<12} {sb:<10.4f} {su:<12.4f} {sp:<10.4f}")

    print(f"\n  Total: {(time.time()-t0)/60:.1f} min")
    print(f"\n  FILES GENERATED:")
    print(f"    final_none.png          — 5-panel (GT,MEEP,BIM,U-Net,PINN)")
    print(f"    final_mild.png          — 5-panel mild")
    print(f"    final_moderate.png      — 5-panel moderate")
    print(f"    final_severe.png        — 5-panel severe")
    print(f"    final_all_levels.png    — 4×4 grid (GT,BIM,U-Net,PINN × 4 levels)")
    print(f"    final_detection_pinn.png — PINN: Healthy vs Severe")
    print(f"    final_detection_unet.png — U-Net: Healthy vs Severe")
    print(f"    results_*.npz           — saved data (never lost again)")
    print(f"{'='*60}")

if __name__=='__main__':
    main()
