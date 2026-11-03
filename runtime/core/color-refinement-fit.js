import {render} from './engine.js';
import {resizeSample,correspondenceWeights,fitBilateral} from './same-image-transfer.js';
import {refineColor} from './color-refinement.js';
const clamp=v=>Math.max(0,Math.min(1,v));
// Smooth residual 3D LUT. Fit output colors jointly, not one editor slider at a time.
export function fitResidualLut(input,target,weights,{size=13,iterations=36,progress=()=>{}}={}){
 const n=size**3,values=new Float32Array(n*3),mass=new Float32Array(n),samples=[];
 const bins=new Float32Array(64);
 for(let k=0;k<weights.length;k++)if(k%7!==0&&weights[k]>.1){const i=k*4,r=clamp(input[i]),g=clamp(input[i+1]),b=clamp(input[i+2]);bins[(Math.min(3,r*4|0)*4+Math.min(3,g*4|0))*4+Math.min(3,b*4|0)]+=weights[k];}
 for(let k=0;k<weights.length;k++){
  if(k%7===0||weights[k]<=.1)continue;const i=k*4,v=[input[i],input[i+1],input[i+2]];
  // Do not learn a clipped lookup from out-of-display-range HDR samples.
  if(v.some(x=>x<0||x>1))continue;
  const coords=v.map(x=>x*(size-1)),base=coords.map(Math.floor),f=coords.map((x,j)=>x-base[j]),indices=[],taps=[];
  const bin=(Math.min(3,v[0]*4|0)*4+Math.min(3,v[1]*4|0))*4+Math.min(3,v[2]*4|0),q=weights[k]/Math.sqrt(Math.max(1,bins[bin]));
  for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++){
   const j=(Math.min(size-1,base[2]+z)*size+Math.min(size-1,base[1]+y))*size+Math.min(size-1,base[0]+x),a=(x?f[0]:1-f[0])*(y?f[1]:1-f[1])*(z?f[2]:1-f[2]);indices.push(j);taps.push(a);mass[j]+=q*a;
  }
  samples.push({v,d:v.map((x,c)=>target[i+c]/255-x),indices,taps,q});
 }
 for(let pass=0;pass<iterations;pass++){
  const gradient=new Float32Array(n*3);
  for(const s of samples){const pred=[0,0,0];for(let t=0;t<8;t++)for(let c=0;c<3;c++)pred[c]+=values[s.indices[t]*3+c]*s.taps[t];for(let c=0;c<3;c++){const e=Math.max(-.2,Math.min(.2,s.d[c]-pred[c]));for(let t=0;t<8;t++)gradient[s.indices[t]*3+c]+=s.q*s.taps[t]*e;}}
  for(let j=0;j<n;j++)for(let c=0;c<3;c++)values[j*3+c]+=.7*gradient[j*3+c]/Math.max(.08,mass[j]);
  const next=values.slice();
  for(let z=0;z<size;z++)for(let y=0;y<size;y++)for(let x=0;x<size;x++){const j=(z*size+y)*size+x,ns=[];for(const [dx,dy,dz] of [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])if(x+dx>=0&&x+dx<size&&y+dy>=0&&y+dy<size&&z+dz>=0&&z+dz<size)ns.push(((z+dz)*size+y+dy)*size+x+dx);const anchor=2+Math.min(12,mass[j]*2);for(let c=0;c<3;c++)next[j*3+c]=(values[j*3+c]*anchor+ns.reduce((s,k)=>s+values[k*3+c],0))/(anchor+ns.length+.025);}
  values.set(next);if(pass%6===0)progress({phase:'精细颜色映射',progress:(pass+1)/iterations});
 }
 return {kind:'lut',size,values,strength:1};
}
function mapPixels(p,w,h,value){const out=new Float32Array(p.length),rgb=new Float64Array(3);for(let k=0;k<w*h;k++){const i=k*4;refineColor(value,p[i],p[i+1],p[i+2],(k%w+.5)/w,(Math.floor(k/w)+.5)/h,rgb);out.set(rgb,i);out[i+3]=p[i+3];}return out;}
function scores(p,t,w,h,weights){const sums=new Float64Array(24),mass=new Float64Array(24);let error=0,n=0,clips=0;
 for(let k=0;k<w*h;k++){if(k%7!==0||weights[k]<.1)continue;const i=k*4,r=t[i]/255,g=t[i+1]/255,b=t[i+2]/255,lo=Math.min(r,g,b),hi=Math.max(r,g,b),family=hi-lo<.08?3:hi===r?0:hi===g?1:2,cell=(Math.floor(k/w)<h/2?0:4)+family,light=(.2126*r+.7152*g+.0722*b)<.33?0:(.2126*r+.7152*g+.0722*b)<.67?1:2,j=cell*3+light;
  let e=0;for(let c=0;c<3;c++){e+=(clamp(p[i+c])-t[i+c]/255)**2;clips+=weights[k]*Number((p[i+c]<0&&t[i+c]>3)||(p[i+c]>1&&t[i+c]<252));}sums[j]+=e*weights[k];mass[j]+=weights[k];error+=e*weights[k];n+=weights[k];
 }
 const groups=Array.from(sums,(s,i)=>mass[i]>=4?s/mass[i]:null),active=groups.filter(x=>x!==null);return {groups,mean:active.reduce((a,b)=>a+b,0)/Math.max(1,active.length),error:Math.sqrt(error/Math.max(1,3*n)),clips:clips/Math.max(1,3*n)};
}
export function refineMatchedColors(source,target,w,h,baseline,{valid,progress=()=>{}}={}){
 const a=resizeSample(source,w,h,256),b=resizeSample(target,w,h,256);
 const resizeValid=(nw,nh)=>valid?Uint8Array.from({length:nw*nh},(_,k)=>valid[Math.min(h-1,Math.floor((Math.floor(k/nw)+.5)*h/nh))*w+Math.min(w-1,Math.floor((k%nw+.5)*w/nw))]):null;
 const weights=correspondenceWeights(a.pixels,b.pixels,a.width,a.height,resizeValid(a.width,a.height));
 const params=structuredClone(baseline.params);delete params.colorRefinement;
 const base=render(a.pixels,a.width,a.height,params,{layers:baseline.layers,outputFormat:'float32'});
 if(scores(base,b.pixels,a.width,a.height,weights).error<.5/255)return {...baseline,colorRefinementAccepted:false};
 const global=fitResidualLut(base,b.pixels,weights,{progress}),value={version:1,strength:1,global};
 const local=fitBilateral(mapPixels(base,a.width,a.height,value),b.pixels,a.width,a.height,weights,p=>progress({...p,phase:'局部颜色残差'}));
 const c=resizeSample(source,w,h,384),d=resizeSample(target,w,h,384),cw=correspondenceWeights(c.pixels,d.pixels,c.width,c.height,resizeValid(c.width,c.height));
 const cb=render(c.pixels,c.width,c.height,params,{layers:baseline.layers,outputFormat:'float32'}),before=scores(cb,d.pixels,c.width,c.height,cw);let best=before.mean,chosen=null;
 for(const strength of [.35,.65,1])for(const localStrength of [0,.5,1]){
  const candidate={...value,strength,...(localStrength?{local:{...local,strength:localStrength}}:{})},after=scores(mapPixels(cb,c.width,c.height,candidate),d.pixels,c.width,c.height,cw);
  if(after.mean<best*.995&&after.error<=before.error*1.01+1e-5&&after.clips<=before.clips+.002&&after.groups.every((v,i)=>v===null||before.groups[i]===null||v<=before.groups[i]*1.10+.000015)){chosen=candidate;best=after.mean;}
 }
 if(!chosen)return {...baseline,colorRefinementAccepted:false};
 params.colorRefinement=chosen;
 const out=render(source,w,h,params,{layers:baseline.layers,outputFormat:'float32'}),full=correspondenceWeights(source,target,w,h,valid);
 return {...baseline,params,error:scores(out,target,w,h,full).error,colorRefinementAccepted:true};
}
