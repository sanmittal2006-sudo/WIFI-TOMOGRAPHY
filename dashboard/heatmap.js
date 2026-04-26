// Lung-shaped heatmap using the anatomical lung PNG as mask
const lungImg = new Image();
lungImg.src = 'lungs.png';

// Better inferno colormap (matches matplotlib)
function infernoColor(t) {
  t = Math.max(0, Math.min(1, t));
  if (t < 0.15) return [Math.floor(t/0.15*40), 0, Math.floor(20+t/0.15*60)];
  if (t < 0.35) return [Math.floor(40+((t-0.15)/0.2)*140), Math.floor(((t-0.15)/0.2)*30), Math.floor(80+((t-0.15)/0.2)*40)];
  if (t < 0.55) return [Math.floor(180+((t-0.35)/0.2)*60), Math.floor(30+((t-0.35)/0.2)*70), Math.floor(120-((t-0.35)/0.2)*90)];
  if (t < 0.75) return [Math.floor(240+((t-0.55)/0.2)*15), Math.floor(100+((t-0.55)/0.2)*100), Math.floor(30-((t-0.55)/0.2)*30)];
  return [255, Math.floor(200+((t-0.75)/0.25)*55), Math.floor(((t-0.75)/0.25)*120)];
}

// Check if pixel is inside the lung shape by sampling the lung image alpha
function createLungMask(w, h) {
  const offscreen = document.createElement('canvas');
  offscreen.width = w; offscreen.height = h;
  const ctx = offscreen.getContext('2d');
  ctx.drawImage(lungImg, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;
  // Mask: pixel is "inside lung" if brightness > 20
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = data[i*4], g = data[i*4+1], b = data[i*4+2];
    mask[i] = (r + g + b) > 60 ? 1 : 0;
  }
  return mask;
}

// Pixel is inside anatomical right lung (LEFT half of image — medical convention)
function isRightLungPixel(x, w) { return x < w * 0.5; }

function drawLungHeatmap(canvas, caseId, mode) {
  // Use setTimeout to prevent blocking UI clicks
  setTimeout(() => _drawLungHeatmapSync(canvas, caseId, mode), 0);
}

function _drawLungHeatmapSync(canvas, caseId, mode) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  
  if (!lungImg.complete) {
    lungImg.onload = () => _drawLungHeatmapSync(canvas, caseId, mode);
    return;
  }

  // Draw lung image first to get the mask
  const mask = createLungMask(W, H);
  
  // Create heatmap
  const img = ctx.createImageData(W, H);
  
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = y * W + x;
      const i = idx * 4;
      
      if (!mask[idx]) {
        // Outside lungs - dark background
        img.data[i] = 10; img.data[i+1] = 14; img.data[i+2] = 23; img.data[i+3] = 255;
        continue;
      }
      
      // Inside lung - generate heatmap value
      const nx = x / W, ny = y / H;
      const isRight = isRightLungPixel(x, W);
      let v = 0;
      
      if (mode === 'recon') {
        // BIM: blurry, noisy reconstruction
        v = 0.08 + Math.random() * 0.06;
        v += 0.04 * Math.sin(nx * 25) * Math.cos(ny * 18);
        v += 0.03 * Math.sin((nx+ny) * 15);
        if (isRight && caseId !== 'healthy') {
          const ecx = 0.32, ecy = 0.52;
          const er = {mild: 0.06, moderate: 0.10, severe: 0.16}[caseId] || 0;
          const ed = Math.sqrt((nx-ecx)**2 + (ny-ecy)**2);
          if (ed < er) { v = 0.5 + 0.48 * (1 - ed/er) * (1 - ed/er); v += Math.random() * 0.05; }
          else if (ed < er * 2) v += 0.15 * Math.max(0, 1 - (ed-er)/er);
        }
        const dcx = Math.abs(nx - (isRight ? 0.32 : 0.68));
        const dcy = Math.abs(ny - 0.5);
        v += 0.03 * (1 - Math.sqrt(dcx*dcx + dcy*dcy) * 2);
      } else if (mode === 'unet') {
        // U-Net: much sharper, cleaner edges
        v = 0.04 + Math.random() * 0.02;
        if (isRight && caseId !== 'healthy') {
          const ecx = 0.32, ecy = 0.52;
          const er = {mild: 0.06, moderate: 0.10, severe: 0.16}[caseId] || 0;
          const ed = Math.sqrt((nx-ecx)**2 + (ny-ecy)**2);
          if (ed < er) { v = 0.7 + 0.28 * (1 - ed/er); }
          else if (ed < er * 1.2) v += 0.08 * Math.max(0, 1 - (ed-er)/(er*0.2));
        }
      } else if (mode === 'pinn') {
        // PINN: sharpest, physics-constrained, almost perfect
        v = 0.02 + Math.random() * 0.01;
        if (isRight && caseId !== 'healthy') {
          const ecx = 0.32, ecy = 0.52;
          const er = {mild: 0.06, moderate: 0.10, severe: 0.16}[caseId] || 0;
          const ed = Math.sqrt((nx-ecx)**2 + (ny-ecy)**2);
          if (ed < er * 0.95) v = 0.85 + 0.14 * (1 - ed/er);
          else if (ed < er) v = 0.4;
        }
      } else if (mode === 'uncertainty') {
        v = 0.03 + Math.random() * 0.08;
        // Higher uncertainty near edema
        if (isRight && caseId !== 'healthy') {
          const ecx = 0.32, ecy = 0.52;
          const er = {mild: 0.08, moderate: 0.13, severe: 0.18}[caseId] || 0;
          const ed = Math.sqrt((nx-ecx)**2 + (ny-ecy)**2);
          if (ed < er * 1.5) {
            v = 0.3 + 0.65 * Math.max(0, 1 - ed/er);
            v += Math.random() * 0.06;
          }
        }
        // Edge uncertainty
        v += 0.05 * Math.random();
      }
      
      v = Math.max(0, Math.min(1, v));
      const [r, g, b] = infernoColor(v);
      img.data[i] = r; img.data[i+1] = g; img.data[i+2] = b; img.data[i+3] = 255;
    }
  }
  
  ctx.putImageData(img, 0, 0);
  
  // Draw the lung outline image on top (composited)
  ctx.globalAlpha = 0.4;
  ctx.drawImage(lungImg, 0, 0, W, H);
  ctx.globalAlpha = 1.0;
}

window.LungHeatmap = { drawLungHeatmap, infernoColor };
