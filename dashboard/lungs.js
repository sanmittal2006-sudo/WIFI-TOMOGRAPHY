// Realistic anatomical lung outlines using canvas paths
function drawAnatomicalLungs(ctx, cx, cy, scale, opts={}) {
  const {leftFill, rightFill, strokeColor='rgba(200,220,255,0.5)', lineWidth=1.5, dashed=false} = opts;
  ctx.save(); ctx.translate(cx, cy); ctx.scale(scale, scale);
  
  // Left lung (viewer's left = anatomical right)
  ctx.beginPath();
  ctx.moveTo(-85, -80);
  ctx.bezierCurveTo(-95, -60, -100, -20, -98, 20);
  ctx.bezierCurveTo(-96, 60, -85, 90, -70, 105);
  ctx.bezierCurveTo(-55, 115, -40, 110, -35, 95);
  ctx.bezierCurveTo(-30, 75, -28, 40, -30, 0);
  ctx.bezierCurveTo(-32, -30, -38, -55, -42, -70);
  ctx.bezierCurveTo(-48, -82, -60, -88, -75, -85);
  ctx.closePath();
  if(leftFill){ctx.fillStyle=leftFill; ctx.fill();}
  ctx.strokeStyle=strokeColor; ctx.lineWidth=lineWidth/scale;
  if(dashed) ctx.setLineDash([4/scale,3/scale]);
  ctx.stroke(); ctx.setLineDash([]);

  // Right lung (viewer's right = anatomical left)
  ctx.beginPath();
  ctx.moveTo(85, -80);
  ctx.bezierCurveTo(95, -60, 100, -20, 98, 20);
  ctx.bezierCurveTo(96, 60, 85, 90, 70, 105);
  ctx.bezierCurveTo(55, 115, 40, 110, 35, 95);
  ctx.bezierCurveTo(30, 75, 28, 40, 30, 0);
  ctx.bezierCurveTo(32, -30, 38, -55, 42, -70);
  ctx.bezierCurveTo(48, -82, 60, -88, 75, -85);
  ctx.closePath();
  if(rightFill){ctx.fillStyle=rightFill; ctx.fill();}
  ctx.strokeStyle=strokeColor; ctx.lineWidth=lineWidth/scale;
  if(dashed) ctx.setLineDash([4/scale,3/scale]);
  ctx.stroke(); ctx.setLineDash([]);

  // Trachea/bronchi
  ctx.strokeStyle='rgba(200,220,255,0.25)'; ctx.lineWidth=1.2/scale;
  ctx.beginPath(); ctx.moveTo(0,-110); ctx.lineTo(0,-70);
  ctx.moveTo(0,-70); ctx.quadraticCurveTo(-15,-55,-30,-45);
  ctx.moveTo(0,-70); ctx.quadraticCurveTo(15,-55,30,-45);
  ctx.stroke();

  ctx.restore();
}

// Draw bottle contour inside right lung
function drawBottleContour(ctx, cx, cy, scale, size, color, dashed=true) {
  if(!size) return;
  ctx.save(); ctx.translate(cx, cy); ctx.scale(scale, scale);
  ctx.strokeStyle = color; ctx.lineWidth = 2/scale;
  if(dashed) ctx.setLineDash([5/scale, 3/scale]);
  ctx.beginPath();
  ctx.ellipse(-65, 15, size*0.6, size, 0.1, 0, Math.PI*2);
  ctx.stroke(); ctx.setLineDash([]);
  ctx.restore();
}

// Draw edema fill in right lung
function drawEdemaFill(ctx, cx, cy, scale, size, color) {
  if(!size) return;
  ctx.save(); ctx.translate(cx, cy); ctx.scale(scale, scale);
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.ellipse(-65, 15, size*0.55, size*0.9, 0.1, 0, Math.PI*2);
  ctx.fill();
  ctx.restore();
}

window.LungDraw = { drawAnatomicalLungs, drawBottleContour, drawEdemaFill };
