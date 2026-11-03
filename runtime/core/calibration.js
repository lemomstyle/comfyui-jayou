// Creative RGB primary calibration in scene-linear RGB. This is not an Adobe
// DCP implementation. Rotate each primary's chroma and scale its amplitude;
// neutral-axis subtraction preserves greys, and reconstruction preserves Y.
export const calibrationControls = ['Red','Green','Blue'].flatMap((name,i)=>[
 [`calibration${name}Hue`,['红','绿','蓝'][i]+'原色色相',-100,100],
 [`calibration${name}Saturation`,['红','绿','蓝'][i]+'原色饱和度',-100,100]
]);
export const calibrationDefaults={...Object.fromEntries(calibrationControls.map(([k])=>[k,0])),calibrationShadowTint:0};
const weights=[.2126,.7152,.0722];
export function calibrationMatrix(p){
 if(calibrationControls.every(([k])=>!p[k]))return null;
 const m=[1,0,0,0,1,0,0,0,1];
 for(let i=0;i<3;i++){
  const name=['Red','Green','Blue'][i],angle=(p[`calibration${name}Hue`]||0)*Math.PI/600,scale=1+(p[`calibration${name}Saturation`]||0)/100;
  const y=weights[i],u=(i===2?1:0)-y,v=(i===0?1:0)-y;
  const du=scale*(u*Math.cos(angle)+v*Math.sin(angle))-u,dv=scale*(v*Math.cos(angle)-u*Math.sin(angle))-v;
  const delta=[dv,(-weights[0]*dv-weights[2]*du)/weights[1],du];
  for(let row=0;row<3;row++)for(let col=0;col<3;col++)m[row*3+col]+=delta[row]*((col===i?1:0)-weights[col]);
 }
 return m;
}

// Shadow tint is a luminance-preserving green/magenta correction. It tapers
// smoothly to zero at linear Y=.25 and does not lift absolute black.
export function shadowCalibration(r,g,b,tint){
 const y=weights[0]*r+weights[1]*g+weights[2]*b;
 if(!tint||y<=0||y>=.25)return [r,g,b];
 const t=y/.25,falloff=1-t*t*(3-2*t),delta=y*(tint/100)*.35*falloff;
 return [r+delta,g-delta*(weights[0]+weights[2])/weights[1],b+delta];
}
