import {boxMean} from './match-regions.js';
const clamp=v=>Math.max(0,Math.min(1,v));
// Tiled, image-guided alpha refinement. Only uncertain boundary pixels change.
export function refineCompensationMask(mask,guide,w,h,{stroke=null,mode='refine',radius=.025}={}){
 if(!(mask instanceof Uint8Array)||mask.length!==w*h||guide.length!==w*h*4)throw Error('修边图像尺寸不一致');
 const result=mask.slice(),brush=stroke?.length?stroke:null,R=Math.max(1,radius*Math.max(w,h));
 const bounds=brush?[Math.min(...brush.map(p=>p[0]*w))-R,Math.min(...brush.map(p=>p[1]*h))-R,Math.max(...brush.map(p=>p[0]*w))+R,Math.max(...brush.map(p=>p[1]*h))+R]:null;
 const influence=(x,y)=>{if(!brush)return 1;if(x<bounds[0]||y<bounds[1]||x>bounds[2]||y>bounds[3])return 0;let d=Infinity;for(let i=0;i<brush.length;i++){const a=brush[Math.max(0,i-1)],b=brush[i],ax=a[0]*w,ay=a[1]*h,dx=b[0]*w-ax,dy=b[1]*h-ay,t=clamp(((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy||1));d=Math.min(d,Math.hypot(x-ax-t*dx,y-ay-t*dy));}return clamp((R-d)/Math.max(1,R*.3));};
 if(brush&&mode!=='refine'){for(let y=0;y<h;y++)for(let x=0;x<w;x++){const v=influence(x+.5,y+.5);if(v){const k=y*w+x;result[k]=Math.round(mask[k]*(1-v)+(mode==='add'?255:0)*v);}}return result;}
 const tile=256,halo=20,r=4;
 for(let ty=0;ty<h;ty+=tile)for(let tx=0;tx<w;tx+=tile){
  if(bounds&&(tx>bounds[2]||ty>bounds[3]||tx+tile<bounds[0]||ty+tile<bounds[1]))continue;
  const left=Math.max(0,tx-halo),top=Math.max(0,ty-halo),right=Math.min(w,tx+tile+halo),bottom=Math.min(h,ty+tile+halo),tw=right-left,th=bottom-top,n=tw*th;
  const I=new Float32Array(n),P=new Float32Array(n),II=new Float32Array(n),IP=new Float32Array(n);
  for(let y=0;y<th;y++)for(let x=0;x<tw;x++){const k=y*tw+x,j=(y+top)*w+x+left;I[k]=(.2126*guide[j*4]+.7152*guide[j*4+1]+.0722*guide[j*4+2])/255;P[k]=mask[j]/255;II[k]=I[k]*I[k];IP[k]=I[k]*P[k];}
  const mi=boxMean(I,tw,th,r),mp=boxMean(P,tw,th,r),mii=boxMean(II,tw,th,r),mip=boxMean(IP,tw,th,r),a=new Float32Array(n),b=new Float32Array(n),band=boxMean(P,tw,th,8);
  for(let k=0;k<n;k++){a[k]=(mip[k]-mi[k]*mp[k])/(Math.max(0,mii[k]-mi[k]*mi[k])+.0004);b[k]=mp[k]-a[k]*mi[k];}
  const ma=boxMean(a,tw,th,r),mb=boxMean(b,tw,th,r);
  for(let y=ty;y<Math.min(h,ty+tile);y++)for(let x=tx;x<Math.min(w,tx+tile);x++){const k=(y-top)*tw+x-left,j=y*w+x;if(band[k]<.002||band[k]>.998)continue;const amount=influence(x+.5,y+.5);if(amount)result[j]=Math.round(mask[j]*(1-amount)+255*clamp(ma[k]*I[k]+mb[k])*amount);}
 }
 return result;
}
export function compensationToneMask(mask,w,h){return Uint8Array.from(boxMean(mask,w,h,Math.max(2,Math.round(Math.max(w,h)/320))),v=>Math.round(v));}
