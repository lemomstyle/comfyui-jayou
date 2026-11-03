import {exteriorBackground,fillSmallHoles} from './mask-topology.js';
// Independent RGB-guided edge reconstruction. No model code or weights are included.
import {adjustMaskEdges} from './paired-selection.js';
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
export function validateEdgeOptions(o={}){
 const mode=o.mode??'none',expand=o.expand??0,feather=o.feather??0,fillHoles=o.fillHoles??false,holeArea=o.holeArea??64;
 if(!['none','hard','soft','adaptive'].includes(mode)||!Number.isFinite(expand)||Math.abs(expand)>12||!Number.isFinite(feather)||feather<0||feather>12)throw Error('边缘设置无效');
 if(typeof fillHoles!=='boolean'||!Number.isFinite(holeArea)||holeArea<1||holeArea>4096)throw Error('孔洞修复参数无效');
 return {mode,expand,feather,fillHoles,holeArea};
}
// Change only uncertain boundaries; flat interiors stay intact. RGB distance also
// separates equal-luminance edges which a luminance-only filter cannot distinguish.
export function guidedMaskEdges(mask,guide,w,h,mode='adaptive'){
 if(mask.length!==w*h||guide.length!==w*h*4)throw Error('蒙版引导图尺寸不一致');
 if(!['none','hard','soft','adaptive'].includes(mode))throw Error('未知边缘类型');
 if(mode==='none')return mask.slice();
 const out=mask.slice(),r=3,exterior=exteriorBackground(mask,w,h);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
  const k=y*w+x,i=k*4;let touchesExterior=false,min=255,max=0,mean=[0,0,0],square=0,count=0;
  for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){
   const j=clamp(y+dy,0,h-1)*w+clamp(x+dx,0,w-1);touchesExterior ||= !!exterior[j];min=Math.min(min,mask[j]);max=Math.max(max,mask[j]);
  }
  if(max-min<2||!touchesExterior||mask[k]<128&&!exterior[k])continue;
  for(let dy=-r;dy<=r;dy+=2)for(let dx=-r;dx<=r;dx+=2){const j=(clamp(y+dy,0,h-1)*w+clamp(x+dx,0,w-1))*4;for(let c=0;c<3;c++){const v=guide[j+c]/255;mean[c]+=v;square+=v*v;}count++;}
  const variance=Math.max(0,square/count-mean.reduce((s,v)=>s+(v/count)**2,0)),sigma=Math.max(.0009,variance*.25);
  let sum=0,mass=0;
  for(let dy=-r;dy<=r;dy++)for(let dx=-r;dx<=r;dx++){
   const j=clamp(y+dy,0,h-1)*w+clamp(x+dx,0,w-1),q=j*4;let d=0;for(let c=0;c<3;c++)d+=((guide[i+c]-guide[q+c])/255)**2;
   const weight=Math.exp(-(dx*dx+dy*dy)/8-d/(2*sigma));sum+=weight*mask[j];mass+=weight;
  }
  let alpha=sum/Math.max(mass,1e-9)/255;
  // Soft mode retains fractional coverage; adaptive uses conservative sharpening
  // only at strong RGB edges, never thresholds an entire sky/foliage mask.
  const hard=mode==='hard'?1:mode==='adaptive'?clamp((variance-.012)/.08)*.5:0;
  const sharpen=clamp((alpha-.5)*1.6+.5);alpha+=hard*(sharpen-alpha);out[k]=Math.round(alpha*255);
  // Edge refinement must not create new background pixels inside the selection.
  if(mask[k]>=128)out[k]=Math.max(128,out[k]);
 }
 return out;
}
export function processMaskEdges(mask,guide,w,h,originalWidth,originalHeight,options={}){
 const o=validateEdgeOptions(options),guided=guidedMaskEdges(mask,guide,w,h,o.mode);
 const filled=o.fillHoles?fillSmallHoles(guided,w,h,Math.floor(o.holeArea*w*h/(originalWidth*originalHeight))):guided;
 return adjustMaskEdges(filled,w,h,originalWidth,originalHeight,o.expand,o.feather);
}
// Local positive/negative clicks: connected color selection inside a radius or
// optional box. This is a deterministic correction tool, not SAM semantic inference.
export function pointSelection(mask,guide,w,h,{point,mode='point-add',radius=.1,tolerance=.12,box=null}={}){
 if(mask.length!==w*h||guide.length!==w*h*4||!Array.isArray(point)||point.length!==2||point.some(v=>!Number.isFinite(v)||v<0||v>1)||!['point-add','point-erase'].includes(mode)||!Number.isFinite(radius)||radius<=0||radius>.5||!Number.isFinite(tolerance)||tolerance<.01||tolerance>.5)throw Error('点选参数无效');
 if(box&&(!Array.isArray(box)||box.length!==4||box.some(v=>!Number.isFinite(v)||v<0||v>1)||box[2]<=box[0]||box[3]<=box[1]))throw Error('点选限制框无效');
 const sx=Math.min(w-1,Math.floor(point[0]*w)),sy=Math.min(h-1,Math.floor(point[1]*h)),R=radius*Math.max(w,h);
 const left=Math.max(0,Math.floor(sx-R),box?Math.floor(box[0]*w):0),right=Math.min(w-1,Math.ceil(sx+R),box?Math.ceil(box[2]*w)-1:w-1),top=Math.max(0,Math.floor(sy-R),box?Math.floor(box[1]*h):0),bottom=Math.min(h-1,Math.ceil(sy+R),box?Math.ceil(box[3]*h)-1:h-1);
 const result=mask.slice();if(sx<left||sx>right||sy<top||sy>bottom)return result;
 const bw=right-left+1,bh=bottom-top+1,seen=new Uint8Array(bw*bh),queue=new Int32Array(bw*bh),seed=(sy*w+sx)*4;let begin=0,end=1;queue[0]=(sy-top)*bw+sx-left;seen[queue[0]]=1;
 while(begin<end){const q=queue[begin++],x=q%bw+left,y=Math.floor(q/bw)+top,i=(y*w+x)*4;let d=0;for(let c=0;c<3;c++)d+=((guide[i+c]-guide[seed+c])/255)**2;d=Math.sqrt(d/3);
  if(d>tolerance||Math.hypot(x-sx,y-sy)>R)continue;
  const a=clamp((tolerance-d)/Math.max(.01,tolerance*.25))*clamp((R-Math.hypot(x-sx,y-sy))/Math.max(1,R*.1)),k=y*w+x;
  result[k]=mode==='point-add'?Math.max(mask[k],Math.round(a*255)):Math.min(mask[k],Math.round((1-a)*255));
  for(const [nx,ny] of [[x-1,y],[x+1,y],[x,y-1],[x,y+1]])if(nx>=left&&nx<=right&&ny>=top&&ny<=bottom){const j=(ny-top)*bw+nx-left;if(!seen[j]){seen[j]=1;queue[end++]=j;}}
 }
 return result;
}
