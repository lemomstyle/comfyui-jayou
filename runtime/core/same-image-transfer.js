import {colorRangeWeight} from './color-range.js';
import {gradientWeight} from './gradients.js';
import {reviewLocalChange,localPenalty} from './local-fit-review.js';
// Fit transformations, never copy reference texture into the source.
import {defaults,render,ranges} from './engine.js';
import {alignPair as coarseAlignment} from './same-image-fit.js';
import {matchColor} from './same-image-map.js';
const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
const scaleOf=p=>p instanceof Float32Array?1:p instanceof Uint16Array?65535:255;
const lum=(p,i)=>.2126*p[i]+.7152*p[i+1]+.0722*p[i+2];
export function resizeSample(p,w,h,max){
 const ratio=Math.min(1,max/Math.max(w,h)),width=Math.max(1,Math.round(w*ratio)),height=Math.max(1,Math.round(h*ratio)),out=new p.constructor(width*height*4);
 // Area averaging preserves sampling consistency across 8/16/float input.
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const x0=x*w/width,x1=(x+1)*w/width,y0=y*h/height,y1=(y+1)*h/height,area=(x1-x0)*(y1-y0),i=(y*width+x)*4,v=[0,0,0,0];
  for(let yy=Math.floor(y0);yy<Math.ceil(y1);yy++)for(let xx=Math.floor(x0);xx<Math.ceil(x1);xx++){
   const weight=(Math.min(xx+1,x1)-Math.max(xx,x0))*(Math.min(yy+1,y1)-Math.max(yy,y0))/area,j=(yy*w+xx)*4;
   for(let c=0;c<4;c++)v[c]+=p[j+c]*weight;
  }for(let c=0;c<4;c++)out[i+c]=v[c];
 }return {pixels:out,width,height};
}
export function correspondenceWeights(source,target,w,h,valid){
 const weights=new Float32Array(w*h),s=scaleOf(source),normalized=Float32Array.from(source,v=>v/s*255);
 // A structural confidence heuristic, not a semantic segmentation claim.
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
  const k=y*w+x;if(valid&&!valid[k])continue;weights[k]=1;if(x<2||y<2||x>=w-2||y>=h-2)continue;
  const i=k*4,dx=8,dy=w*8,ax=lum(normalized,i+dx)-lum(normalized,i-dx),ay=lum(normalized,i+dy)-lum(normalized,i-dy),bx=lum(target,i+dx)-lum(target,i-dx),by=lum(target,i+dy)-lum(target,i-dy),a=Math.hypot(ax,ay),b=Math.hypot(bx,by);
  if(Math.max(a,b)>16){const cosine=(ax*bx+ay*by)/(a*b+1e-6);if(Math.min(a,b)<2||cosine<.35)weights[k]=.05;else if(cosine<.75)weights[k]=.35;}
  // Clipped target pixels do not identify an invertible color relationship.
  if(target[i]>252&&target[i+1]>252&&target[i+2]>252)weights[k]*=.15;
 }return weights;
}
function loss(a,b,weights,holdout=false){let total=0,n=0;for(let k=0;k<weights.length;k++){if((k%7===0)!==holdout)continue;const weight=weights[k];if(!weight)continue;for(let c=0;c<3;c++){const d=Math.abs(a[k*4+c]-b[k*4+c])/255;total+=weight*(d<.06?d*d:.12*d-.0036);n+=weight;}}return Math.sqrt(total/Math.max(1,n));}
function rmse(a,b,weights,holdout=false){let total=0,n=0;for(let k=0;k<weights.length;k++){if((k%7===0)!==holdout)continue;for(let c=0;c<3;c++){total+=weights[k]*((a[k*4+c]-b[k*4+c])/255)**2;n+=weights[k];}}return Math.sqrt(total/Math.max(1,n));}
function descriptors(params,full){
 const keys=['exposure','temperature','tint','contrast','shadows','highlights','whites','blacks','saturation','vibrance'];
 const list=keys.map(key=>({get:()=>params[key],set:v=>params[key]=v,min:ranges[key][0],max:ranges[key][1],step:key==='exposure'?.55:14}));
 if(full){for(const band of params.hsl)for(const key of ['hue','saturation','lightness'])list.push({get:()=>band[key],set:v=>band[key]=v,min:key==='hue'?-40:-70,max:key==='hue'?40:70,step:10});
 for(let i=1;i<4;i++)list.push({get:()=>params.curve[i],set:v=>params.curve[i]=clamp(v,params.curve[i-1],params.curve[i+1]),min:0,max:1,step:.045});
 for(const channel of ['red','green','blue']){params.pointCurves[channel]=[0,.25,.5,.75,1].map(x=>({x,y:x}));for(let i=1;i<4;i++)list.push({get:()=>params.pointCurves[channel][i].y,set:v=>params.pointCurves[channel][i].y=clamp(v,params.pointCurves[channel][i-1].y,params.pointCurves[channel][i+1].y),min:0,max:1,step:.025});}}
 return list;
}
function optimize(descriptors,evaluate,passes,progress){let best=evaluate();for(let pass=0;pass<passes;pass++){for(const d of descriptors){const old=d.get(),step=d.step*2**(-pass);let value=old;for(const dir of [-1,1]){d.set(clamp(old+dir*step,d.min,d.max));const e=evaluate();if(e<best-1e-10){best=e;value=d.get();}}d.set(value);}progress?.((pass+1)/passes);}return best;}
export function residualRegions(rendered,target,w,h,weights){
 // Connected regions of similarly directed low-frequency color error.
 const gw=12,gh=12,cells=Array.from({length:gw*gh},()=>({n:0,d:[0,0,0]}));
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){const k=y*w+x,cell=cells[Math.min(gh-1,Math.floor(y/h*gh))*gw+Math.min(gw-1,Math.floor(x/w*gw))],weight=weights[k];cell.n+=weight;for(let c=0;c<3;c++)cell.d[c]+=weight*(target[k*4+c]-rendered[k*4+c])/255;}
 for(const cell of cells){cell.d=cell.d.map(v=>v/Math.max(1,cell.n));const a=cell.d.map(Math.abs),c=a.indexOf(Math.max(...a));cell.tag=a[c]>.018?(c+1)*Math.sign(cell.d[c]):0;cell.energy=a.reduce((a,b)=>a+b,0);}
 const seen=new Set(),groups=[];for(let i=0;i<cells.length;i++){if(seen.has(i)||!cells[i].tag)continue;const stack=[i],group=[];seen.add(i);while(stack.length){const k=stack.pop();group.push(k);for(const j of [k-gw,k+gw,...(k%gw?[k-1]:[]),...(k%gw<gw-1?[k+1]:[])])if(j>=0&&j<cells.length&&!seen.has(j)&&cells[j].tag===cells[i].tag){seen.add(j);stack.push(j);}}if(group.length>=2)groups.push(group);}
 return groups.sort((a,b)=>b.reduce((s,k)=>s+cells[k].energy,0)-a.reduce((s,k)=>s+cells[k].energy,0)).slice(0,4).map((group,index)=>{
 const xs=group.map(k=>k%gw),ys=group.map(k=>Math.floor(k/gw));return {name:`局部色调 ${index+1}`,type:'radial',gradient:{kind:'radial',x1:clamp((Math.min(...xs)-.5)/gw),x2:clamp((Math.max(...xs)+1.5)/gw),y1:clamp((Math.min(...ys)-.5)/gh),y2:clamp((Math.max(...ys)+1.5)/gh),feather:.95,angle:0,aspect:w/h,invert:false},exposure:0,temperature:0,tint:0,contrast:0,saturation:0,opacity:1,enabled:true};
 });
}
// Parameter fitting uses continuous renderer output. An 8-bit objective hides
// small improvements and encourages compensating channel/HSL adjustments.
function parameterLoss(output,target,weights,holdout=false){
 let total=0,mass=0;
 const huber=d=>{d=Math.abs(d);return d<.06?d*d:.12*d-.0036;};
 for(let k=0;k<weights.length;k++){
  if((k%7===0)!==holdout||!weights[k])continue;
  const i=k*4,t=[target[i]/255,target[i+1]/255,target[i+2]/255],o=[clamp(output[i]),clamp(output[i+1]),clamp(output[i+2])],d=o.map((v,c)=>v-t[c]);
  const neutral=1-clamp((Math.max(...t)-Math.min(...t))/.16);
  // Neutral walls/skin highlights must not trade visible chroma for a small
  // luminance gain elsewhere. Colored references remain valid targets.
  total+=weights[k]*(d.reduce((v,x)=>v+huber(x),0)/3+neutral*(huber(d[0]-d[1])+huber(d[2]-d[1])));
  mass+=weights[k];
 }
 return total/Math.max(1,mass);
}
function parameterPenalty(params){
 let penalty=0;
 for(const channel of ['rgb','red','green','blue']){
  const points=params.pointCurves[channel];
  for(let i=1;i<points.length-1;i++){
   const left=(points[i].y-points[i-1].y)/(points[i].x-points[i-1].x),right=(points[i+1].y-points[i].y)/(points[i+1].x-points[i].x);
   penalty+=.000015*(right-left)**2;
  }
 }
 for(let i=0;i<8;i++)for(const key of ['hue','saturation','lightness']){
  penalty+=.000025*(params.hsl[i][key]/40)**2;
  penalty+=.000015*((params.hsl[i][key]-params.hsl[(i+1)%8][key])/40)**2;
 }
 return penalty;
}
export function solveParameters(source,target,w,h,{valid,progress=()=>{}}={}){
 const params=defaults(),all=descriptors(params,true);let layers=[];
 // Use the editable smooth RGB master curve instead of the piecewise-linear
 // five-point curve, whose corners can become visible after strong exposure.
 params.pointCurves.rgb=[0,.25,.5,.75,1].map(x=>({x,y:x}));
 const tone=params.pointCurves.rgb.slice(1,-1).map((_,j)=>{const i=j+1;return {get:()=>params.pointCurves.rgb[i].y,set:v=>params.pointCurves.rgb[i].y=clamp(v,params.pointCurves.rgb[i-1].y+.01,params.pointCurves.rgb[i+1].y-.01),min:0,max:1,step:.05};});
 const basic=all.slice(0,10),hsl=all.slice(10,34),channels=all.slice(37);
 const reviewSource=resizeSample(source,w,h,384),reviewTarget=resizeSample(target,w,h,384),reviewWeights=correspondenceWeights(reviewSource.pixels,reviewTarget.pixels,reviewSource.width,reviewSource.height,valid?resizeWeights(valid,w,h,reviewSource.width,reviewSource.height):null);
 const review=()=>parameterLoss(render(reviewSource.pixels,reviewSource.width,reviewSource.height,params,{layers,outputFormat:'float32'}),reviewTarget.pixels,reviewWeights,true);
 // Validate each family on a larger image, not only neighboring pixels in
 // the same 192px thumbnail. Keep the last genuinely better recipe.
 let bestReview=review(),bestParams=structuredClone(params),bestLayers=[];

 for(const [stage,max,passes] of [[0,96,6],[1,192,4],[2,192,4]]){
  const a=resizeSample(source,w,h,max),b=resizeSample(target,w,h,max),weights=correspondenceWeights(a.pixels,b.pixels,a.width,a.height,valid?resizeWeights(valid,w,h,a.width,a.height):null);
  const evaluate=()=>parameterLoss(render(a.pixels,a.width,a.height,params,{layers,outputFormat:'float32'}),b.pixels,weights)+parameterPenalty(params);
  if(stage===0){let score=evaluate(),exposure=0;for(const value of [-2,-1,1,2,3]){params.exposure=value;const e=evaluate();if(e<score){score=e;exposure=value;}}params.exposure=exposure;}
  // First establish exposure/tone, then channel balance, then selective color.
  // Descriptor access must resolve the current recipe after a checkpoint.
  const groups=[basic,tone,channels,hsl];
  for(let group=0;group<groups.length;group++){
   optimize(groups[group].map(d=>({...d,step:d.step*(stage?.4:1)})),evaluate,passes);
   // HSL descriptors capture band objects, so restore in place below.
   const score=review();if(score<bestReview){bestReview=score;bestParams=structuredClone(params);bestLayers=structuredClone(layers);}
   else{for(const key of Object.keys(params)){if(key==='hsl')params.hsl.forEach((band,i)=>Object.assign(band,bestParams.hsl[i]));else params[key]=structuredClone(bestParams[key]);}layers=structuredClone(bestLayers);}
   progress({phase:stage?'精细参数反求':'全局参数反求',progress:(stage+group/4)/4});
  }
 }
 const a=resizeSample(source,w,h,192),b=resizeSample(target,w,h,192),weights=correspondenceWeights(a.pixels,b.pixels,a.width,a.height,valid?resizeWeights(valid,w,h,a.width,a.height):null);
 const candidates=residualRegions(render(a.pixels,a.width,a.height,params),b.pixels,a.width,a.height,weights);
 for(const l of candidates){
  const before=review(),beforePixels=render(reviewSource.pixels,reviewSource.width,reviewSource.height,params,{layers,outputFormat:'float32'});layers.push(l);
  const evaluate=()=>parameterLoss(render(a.pixels,a.width,a.height,params,{layers,outputFormat:'float32'}),b.pixels,weights)+localPenalty(l);
  const ds=['exposure','temperature','tint','contrast','saturation'].map(key=>({get:()=>l[key],set:v=>l[key]=v,min:key==='exposure'?-.8:key==='contrast'?-20:-18,max:key==='exposure'?.8:key==='contrast'?20:18,step:key==='exposure'?.3:10}));
  optimize(ds,evaluate,5);
  // Fit geometry too, rather than treating the coarse residual rectangle as truth.
  const g=l.gradient,cx=(g.x1+g.x2)/2,cy=(g.y1+g.y2)/2,rx=(g.x2-g.x1)/2,ry=(g.y2-g.y1)/2;
  const geo={x:cx,y:cy,sx:1,sy:1,angle:0,feather:.95};
  const update=()=>Object.assign(g,{x1:geo.x-rx*geo.sx,x2:geo.x+rx*geo.sx,y1:geo.y-ry*geo.sy,y2:geo.y+ry*geo.sy,angle:geo.angle,feather:geo.feather});
  optimize([['x',cx-.06,cx+.06,.03],['y',cy-.06,cy+.06,.03],['sx',.75,1.25,.15],['sy',.75,1.25,.15],['angle',-30,30,15],['feather',.85,1,.05]].map(([key,min,max,step])=>({get:()=>geo[key],set:v=>{geo[key]=v;update();},min,max,step})),evaluate,2);
  let chosen=0,best=before;
  for(const opacity of [.25,.5,.75,1]){l.opacity=opacity;const pixels=render(reviewSource.pixels,reviewSource.width,reviewSource.height,params,{layers,outputFormat:'float32'}),score=parameterLoss(pixels,reviewTarget.pixels,reviewWeights,true)+localPenalty(l);
   if(score<best&&reviewLocalChange(beforePixels,pixels,reviewTarget.pixels,reviewSource.width,reviewSource.height,reviewWeights,g).accepted){chosen=opacity;best=score;}
  }
  l.opacity=chosen;if(!chosen)layers.pop();
 }
 for(const c of ['rgb','red','green','blue'])if(params.pointCurves[c].every(p=>Math.abs(p.x-p.y)<1e-8))params.pointCurves[c]=[{x:0,y:0},{x:1,y:1}];
 progress({phase:'高分辨率复核',progress:1});const finalWeights=correspondenceWeights(source,target,w,h,valid),base=render(source,w,h,defaults()),out=render(source,w,h,params,{layers});
 return {params,layers,initial:rmse(base,target,finalWeights,true),error:rmse(out,target,finalWeights,true),excluded:1-Array.from(finalWeights).filter(v=>v>=.5).length/finalWeights.length};
}
function resizeWeights(p,w,h,nw,nh){const out=new Uint8Array(nw*nh);for(let y=0;y<nh;y++)for(let x=0;x<nw;x++)out[y*nw+x]=p[Math.min(h-1,Math.floor((y+.5)*h/nh))*w+Math.min(w-1,Math.floor((x+.5)*w/nw))];return out;}
function solve4(A,b){const rows=Array.from({length:4},(_,i)=>[...A.slice(i*4,i*4+4),b[i]]);for(let i=0;i<4;i++){let pivot=i;for(let k=i+1;k<4;k++)if(Math.abs(rows[k][i])>Math.abs(rows[pivot][i]))pivot=k;[rows[i],rows[pivot]]=[rows[pivot],rows[i]];if(Math.abs(rows[i][i])<1e-10)return null;const d=rows[i][i];for(let j=i;j<5;j++)rows[i][j]/=d;for(let k=0;k<4;k++)if(k!==i){const a=rows[k][i];for(let j=i;j<5;j++)rows[k][j]-=a*rows[i][j];}}return rows.map(r=>r[4]);}
export function fitBilateral(source,target,w,h,weights,progress=()=>{}){
 const scale=scaleOf(source),width=5,height=5,depth=8,n=width*height*depth;let guideMax=1;for(let i=0;i<source.length;i+=4)guideMax=Math.max(guideMax,lum(source,i)/scale);guideMax=Math.min(16,guideMax);
 // A common fitted prior keeps sparse luminance cells on the same color
 // transform as populated cells, rather than pulling them back to identity.
 const globalA=new Float64Array(16),globalB=new Float64Array(12);let globalMass=0;
 for(let k=0;k<w*h;k++){if(k%7===0||weights[k]<.1)continue;const i=k*4,v=[source[i]/scale,source[i+1]/scale,source[i+2]/scale,1],q=weights[k];globalMass+=q;for(let a=0;a<4;a++){for(let b=0;b<4;b++)globalA[a*4+b]+=q*v[a]*v[b];for(let c=0;c<3;c++)globalB[c*4+a]+=q*v[a]*target[i+c]/255;}}
 const globalRidge=Math.max(.01,globalMass*.0001),prior=new Float64Array(12);
 for(let a=0;a<4;a++)globalA[a*4+a]+=globalRidge;
 for(let c=0;c<3;c++){const b=globalB.slice(c*4,c*4+4);b[c]+=globalRidge;const solved=solve4(globalA,b);for(let a=0;a<4;a++)prior[c*4+a]=solved?solved[a]:Number(a===c);}
 const values=new Float32Array(n*12);for(let i=0;i<n;i++)values.set(prior,i*12);
 const map={kind:'bilateral',width,height,depth,guideMax,values,strength:1},rgb=new Float64Array(3);
 for(let pass=0;pass<3;pass++){
  const matrices=new Float64Array(n*16),rhs=new Float64Array(n*12),mass=new Float64Array(n);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
   const k=y*w+x;if(k%7===0||weights[k]<.1)continue;const i=k*4,v=[source[i]/scale,source[i+1]/scale,source[i+2]/scale,1];let weight=weights[k];
   if(pass){matchColor(map,...v.slice(0,3),(x+.5)/w,(y+.5)/h,rgb);const error=Math.sqrt([0,1,2].reduce((s,c)=>s+(rgb[c]-target[i+c]/255)**2,0)/3);weight*=Math.min(1,.04/Math.max(.0001,error));}
   const coords=[clamp((x+.5)/w*width-.5,0,width-1),clamp((y+.5)/h*height-.5,0,height-1),clamp(lum(source,i)/scale/guideMax)*(depth-1)],base=coords.map(Math.floor),f=coords.map((v,i)=>v-base[i]);
   for(let z=0;z<2;z++)for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++){
    const j=(Math.min(depth-1,base[2]+z)*height+Math.min(height-1,base[1]+dy))*width+Math.min(width-1,base[0]+dx),q=weight*(dx?f[0]:1-f[0])*(dy?f[1]:1-f[1])*(z?f[2]:1-f[2]);mass[j]+=q;
    for(let a=0;a<4;a++){for(let b=0;b<4;b++)matrices[j*16+a*4+b]+=q*v[a]*v[b];for(let c=0;c<3;c++)rhs[j*12+c*4+a]+=q*v[a]*target[i+c]/255;}
   }
  }
  for(let j=0;j<n;j++){
   const A=matrices.subarray(j*16,j*16+16),ridge=Math.max(2,mass[j]*.04);for(let a=0;a<4;a++)A[a*4+a]+=ridge;
   for(let c=0;c<3;c++){const b=rhs.slice(j*12+c*4,j*12+c*4+4);for(let a=0;a<4;a++)b[a]+=ridge*prior[c*4+a];const solution=solve4(A,b);if(solution)for(let a=0;a<4;a++)values[j*12+c*4+a]=solution[a];}
  }
  // Smooth transformation coefficients, not image pixels. Tether populated cells.
  for(let iter=0;iter<6;iter++){const next=values.slice();for(let z=0;z<depth;z++)for(let y=0;y<height;y++)for(let x=0;x<width;x++){const j=(z*height+y)*width+x,neighbors=[];for(const [dx,dy,dz] of [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])if(x+dx>=0&&x+dx<width&&y+dy>=0&&y+dy<height&&z+dz>=0&&z+dz<depth)neighbors.push(((z+dz)*height+y+dy)*width+x+dx);const anchor=Math.max(1,Math.min(30,mass[j]));for(let a=0;a<12;a++)next[j*12+a]=(values[j*12+a]*anchor+neighbors.reduce((s,k)=>s+values[k*12+a],0))/(anchor+neighbors.length);}values.set(next);}
  progress({phase:'保边颜色变换',progress:(pass+1)/3});
 }
 return map;
}
export function solveColorTransfer(source,target,w,h,{valid,progress=()=>{}}={}){
 const weights=correspondenceWeights(source,target,w,h,valid),map=fitBilateral(source,target,w,h,weights,progress),params={...defaults(),matchMap:map},base=render(source,w,h,defaults()),out=render(source,w,h,params);
 return {params,layers:[],initial:rmse(base,target,weights,true),error:rmse(out,target,weights,true),excluded:1-Array.from(weights).filter(v=>v>=.5).length/weights.length};
}
function at(p,w,h,x,y,c){x=clamp(x,0,w-1);y=clamp(y,0,h-1);const ix=Math.floor(x),iy=Math.floor(y),fx=x-ix,fy=y-iy;return p[(iy*w+ix)*4+c]*(1-fx)*(1-fy)+p[(iy*w+Math.min(ix+1,w-1))*4+c]*fx*(1-fy)+p[(Math.min(iy+1,h-1)*w+ix)*4+c]*(1-fx)*fy+p[(Math.min(iy+1,h-1)*w+Math.min(ix+1,w-1))*4+c]*fx*fy;}
export function alignPair(a,b,w,h){
 let best=coarseAlignment(a,b,w,h,8);
 const gray=(p,x,y)=>.2126*at(p,w,h,x,y,0)+.7152*at(p,w,h,x,y,1)+.0722*at(p,w,h,x,y,2);
 const score=(dx,dy)=>{let sum=0,n=0;for(let y=2;y<h-2;y+=2)for(let x=2;x<w-2;x+=2){const xx=x+dx,yy=y+dy;if(xx<1||yy<1||xx>=w-1||yy>=h-1)continue;const ax=gray(a,x+1,y)-gray(a,x-1,y),ay=gray(a,x,y+1)-gray(a,x,y-1),bx=gray(b,xx+1,yy)-gray(b,xx-1,yy),by=gray(b,xx,yy+1)-gray(b,xx,yy-1),na=Math.hypot(ax,ay),nb=Math.hypot(bx,by);if(na>3&&nb>3){sum+=(ax*bx+ay*by)/(na*nb);n++;}}return n>=Math.max(12,w*h*.003)?sum/n:0;};
 best.score=score(best.dx,best.dy);for(const step of [.5,.25]){const origin={...best};for(const dy of [-step,0,step])for(const dx of [-step,0,step]){const value=score(origin.dx+dx,origin.dy+dy);if(value>best.score+.0005)best={dx:origin.dx+dx,dy:origin.dy+dy,score:value};}}
 return {...best,x:best.dx/w,y:best.dy/h};
}
export function alignedTarget(source,target,w,h,alignment){const pixels=new Uint8ClampedArray(source),valid=new Uint8Array(w*h),dx=alignment.x*w,dy=alignment.y*h;for(let y=0;y<h;y++)for(let x=0;x<w;x++){const xx=x+dx,yy=y+dy;if(xx<0||yy<0||xx>w-1||yy>h-1)continue;const i=(y*w+x)*4;valid[i/4]=1;for(let c=0;c<4;c++)pixels[i+c]=at(target,w,h,xx,yy,c);}return {pixels,valid};}

// Opt-in refinement only: never change solveParameters' stable baseline.
export function enhanceParameters(source,target,w,h,baseline,{valid,progress=()=>{}}={}){
 const params=structuredClone(baseline.params),layers=structuredClone(baseline.layers),a=resizeSample(source,w,h,192),b=resizeSample(target,w,h,192),weights=correspondenceWeights(a.pixels,b.pixels,a.width,a.height,valid?resizeWeights(valid,w,h,a.width,a.height):null);
 function objective(out,t,ws){
  let sum=0,n=0;const bins=Array.from({length:16},()=>({n:0,a:0,b:0,aa:0,bb:0}));
  for(let k=0;k<ws.length;k++){if(!ws[k])continue;const i=k*4,r=clamp(out[i]),g=clamp(out[i+1]),bl=clamp(out[i+2]),tr=t[i]/255,tg=t[i+1]/255,tb=t[i+2]/255,y=.2126*r+.7152*g+.0722*bl,ty=.2126*tr+.7152*tg+.0722*tb,ch=Math.max(tr,tg,tb)-Math.min(tr,tg,tb),q=ws[k]*(1+2*clamp(ch/.15)),neutral=1-clamp(ch/.06);
   sum+=q*((y-ty)**2+(2+18*neutral)*((r-g-tr+tg)**2+(bl-g-tb+tg)**2));
   // Penalize new clipping relative to this reference, including one-channel clipping.
   for(let c=0;c<3;c++){const v=out[i+c],tv=t[i+c]/255;sum+=q*3*(Math.max(0,v-Math.max(.97,tv+.02))**2+Math.max(0,Math.min(.03,tv-.02)-v)**2);}
   n+=q;const cell=bins[Math.min(15,Math.floor(ty*16))];cell.n+=ws[k];cell.a+=ws[k]*y;cell.b+=ws[k]*ty;cell.aa+=ws[k]*y*y;cell.bb+=ws[k]*ty*ty;
  }
  let tone=0,count=0;for(const cell of bins)if(cell.n>8){tone+=(cell.a/cell.n-cell.b/cell.n)**2;count++;}
  return sum/Math.max(1,n)+.5*tone/Math.max(1,count)+parameterPenalty(params);
 }
 const evaluate=()=>objective(render(a.pixels,a.width,a.height,params,{layers,outputFormat:'float32'}),b.pixels,weights);
 const ds=['exposure','contrast','shadows','highlights','whites','blacks','temperature','tint','saturation','vibrance'].map(key=>({get:()=>params[key],set:v=>params[key]=v,min:ranges[key][0],max:ranges[key][1],step:key==='exposure'?.15:8}));
 for(let i=0;i<8;i++)for(const key of ['hue','saturation'])ds.push({get:()=>params.hsl[i][key],set:v=>params.hsl[i][key]=v,min:-70,max:70,step:key==='hue'?4:8});
 for(let pass=0;pass<2;pass++)optimize(ds,evaluate,4,p=>progress({phase:'色彩与对比增强',progress:(pass+p)/2}));
 const fullWeights=correspondenceWeights(source,target,w,h,valid),before=render(source,w,h,baseline.params,{layers:baseline.layers,outputFormat:'float32'}),after=render(source,w,h,params,{layers,outputFormat:'float32'});
 const accepted=objective(after,target,fullWeights)<objective(before,target,fullWeights)*.995;
 const chosen=accepted?{params,layers}:baseline,out=render(source,w,h,chosen.params,{layers:chosen.layers});
 return {...baseline,params:chosen.params,layers:chosen.layers,error:rmse(out,target,fullWeights,true),enhancementAccepted:accepted};
}

// Optional, editable color/spatial corrections. No reference pixels or raster masks.
export function refineRegionalParameters(source,target,w,h,baseline,{valid,progress=()=>{}}={}){
 const params=structuredClone(baseline.params),layers=structuredClone(baseline.layers);
 const a=resizeSample(source,w,h,160),b=resizeSample(target,w,h,160);
 const weights=correspondenceWeights(a.pixels,b.pixels,a.width,a.height,valid?resizeWeights(valid,w,h,a.width,a.height):null);
 const check=resizeSample(source,w,h,384),goal=resizeSample(target,w,h,384);
 const cw=correspondenceWeights(check.pixels,goal.pixels,check.width,check.height,valid?resizeWeights(valid,w,h,check.width,check.height):null);
 const candidates=[];
 for(const [hue,label] of [[30,'暖色'],[90,'黄绿'],[150,'绿色'],[210,'青蓝'],[270,'蓝紫'],[330,'红色']])for(const lower of [false,true]){
  candidates.push({name:`分区精调 · ${label}${lower?'下部':'上部'}`,type:'linear',enabled:true,opacity:1,
   gradient:{kind:'linear',x1:0,y1:.15,x2:0,y2:.85,feather:1,invert:lower,aspect:w/h},
   colorRange:{hue,width:20,feather:40,satMin:.04,satMax:1},exposure:0,temperature:0,tint:0,contrast:0,saturation:0});
 }
 function regionWeights(p,pw,ph,base,l){const scale=scaleOf(p);return Float32Array.from(base,(q,k)=>q*gradientWeight(l.gradient,(k%pw+.5)/pw,(Math.floor(k/pw)+.5)/ph)*colorRangeWeight(l.colorRange,p[k*4]/scale,p[k*4+1]/scale,p[k*4+2]/scale));}
 const train=candidates.map(l=>regionWeights(a.pixels,a.width,a.height,weights,l));
 const review=candidates.map(l=>regionWeights(check.pixels,check.width,check.height,cw,l));
 const active=candidates.map((_,i)=>i).filter(i=>train[i].reduce((s,v)=>s+v,0)>a.width*a.height*.008);
 const score=(pixels,target,sets,holdout)=>active.map(i=>parameterLoss(pixels,target,sets[i],holdout));
 const average=v=>v.reduce((a,b)=>a+b,0)/Math.max(1,v.length);
 const renderCheck=()=>render(check.pixels,check.width,check.height,params,{layers,outputFormat:'float32'});
 let accepted=0;
 for(const index of active){
  const l=candidates[index],beforePixels=renderCheck(),before=score(beforePixels,goal.pixels,review,true);
  layers.push(l);let bestLayer=null,bestScore=average(before);
  // Multiple starts allow exposure and chroma to move together before coordinate descent.
  for(const start of [[0,0,0],[.35,12,15],[-.35,-12,15]]){
   Object.assign(l,{exposure:start[0],temperature:start[1],saturation:start[2],tint:0,contrast:0});
   const ds=['exposure','temperature','tint','contrast','saturation'].map(key=>({get:()=>l[key],set:v=>l[key]=v,min:key==='exposure'?-1.2:key==='saturation'?-45:-30,max:key==='exposure'?1.2:key==='saturation'?50:30,step:key==='exposure'?.3:12}));
   const evaluate=()=>{const out=render(a.pixels,a.width,a.height,params,{layers,outputFormat:'float32'});return .6*parameterLoss(out,b.pixels,train[index])+.4*average(score(out,b.pixels,train,false))+.000002*(l.exposure*l.exposure+(l.saturation/60)**2);};
   optimize(ds,evaluate,4);
   const after=score(renderCheck(),goal.pixels,review,true),value=average(after);
   // A local improvement must not hide degradation in another populated region.
   if(value<bestScore*.998&&after.every((v,i)=>v<=before[i]*1.04+.000002)){
    bestLayer=structuredClone(l);bestScore=value;
   }
  }
  if(bestLayer){Object.assign(l,bestLayer);accepted++;}else layers.pop();
  progress({phase:'分区精细匹配',progress:(active.indexOf(index)+1)/active.length});
 }
 const fullWeights=correspondenceWeights(source,target,w,h,valid),out=render(source,w,h,params,{layers});
 return {...baseline,params,layers,error:rmse(out,target,fullWeights,true),regionalAccepted:accepted>0,regionalCount:accepted};
}
