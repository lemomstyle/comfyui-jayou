import {defaults,render,ranges} from './engine.js';
import {matchColor} from './same-image-map.js';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function luminance(p,i){return .2126*p[i]+.7152*p[i+1]+.0722*p[i+2];}
// Gradient direction correlation is less sensitive to exposure than RGB difference.
export function alignmentScore(a,b,w,h,dx=0,dy=0){let sum=0,n=0;for(let y=2;y<h-2;y++)for(let x=2;x<w-2;x++){const xx=x+dx,yy=y+dy;if(xx<2||xx>=w-2||yy<2||yy>=h-2)continue;const i=(y*w+x)*4,j=(yy*w+xx)*4,ax=luminance(a,i+4)-luminance(a,i-4),ay=luminance(a,i+w*4)-luminance(a,i-w*4),bx=luminance(b,j+4)-luminance(b,j-4),by=luminance(b,j+w*4)-luminance(b,j-w*4),an=Math.hypot(ax,ay),bn=Math.hypot(bx,by);if(an<3||bn<3)continue;sum+=(ax*bx+ay*by)/(an*bn);n++;}return n>=Math.max(20,w*h*.015)?sum/n:0;}
export function alignPair(a,b,w,h,search=5){let best={dx:0,dy:0,score:alignmentScore(a,b,w,h)};for(let dy=-search;dy<=search;dy++)for(let dx=-search;dx<=search;dx++){const score=alignmentScore(a,b,w,h,dx,dy);if(score>best.score+.002)best={dx,dy,score};}return {...best,x:best.dx/w,y:best.dy/h};}
export function alignedTarget(source,target,w,h,alignment){const pixels=new Uint8ClampedArray(source),valid=new Uint8Array(w*h);const dx=Math.round(alignment.x*w),dy=Math.round(alignment.y*h);for(let y=0;y<h;y++)for(let x=0;x<w;x++){const xx=x+dx,yy=y+dy;if(xx<0||yy<0||xx>=w||yy>=h)continue;const i=(y*w+x)*4,j=(yy*w+xx)*4;pixels.set(target.subarray(j,j+4),i);valid[y*w+x]=1;}return {pixels,valid};}
export function colorError(a,b,valid){let e=0,n=0;for(let i=0;i<a.length;i+=4){if(valid&&!valid[i/4])continue;for(let c=0;c<3;c++){const d=(a[i+c]-b[i+c])/255;e+=d*d;n++;}}return Math.sqrt(e/Math.max(1,n));}
export function fitParameters(source,target,w,h,{valid,local=false,progress=()=>{}}={}){
 let params=defaults(),layers=[];const evaluate=()=>colorError(render(source,w,h,params,{layers}),target,valid);let error=evaluate(),initial=error;
 const keys=['exposure','temperature','tint','contrast','shadows','highlights','whites','blacks','saturation','vibrance'];
 for(let pass=0;pass<5;pass++){
  for(const key of keys){const old=params[key],step=(key==='exposure'?.5:16)*2**(-pass),[min,max]=ranges[key];let best=old;for(const dir of [-1,1]){params[key]=clamp(old+dir*step,min,max);const e=evaluate();if(e<error){error=e;best=params[key];}}params[key]=best;}
  for(let band=0;band<8;band++)for(const key of ['hue','saturation','lightness']){const old=params.hsl[band][key],limit=key==='hue'?40:70,step=12*2**(-pass);let best=old;for(const dir of [-1,1]){params.hsl[band][key]=clamp(old+dir*step,-limit,limit);const e=evaluate();if(e<error){error=e;best=params.hsl[band][key];}}params.hsl[band][key]=best;}
  for(let i=1;i<4;i++){const old=params.curve[i],step=.06*2**(-pass);let best=old;for(const dir of [-1,1]){params.curve[i]=clamp(old+dir*step,params.curve[i-1],params.curve[i+1]);const e=evaluate();if(e<error){error=e;best=params.curve[i];}}params.curve[i]=best;}
  progress({phase:'全局参数拟合',progress:(pass+1)/5});
 }
 if(local){
  for(let y=0;y<2;y++)for(let x=0;x<3;x++)layers.push({name:`局部匹配 ${y*3+x+1}`,type:'radial',gradient:{kind:'radial',x1:x/3-.12,y1:y/2-.15,x2:(x+1)/3+.12,y2:(y+1)/2+.15,feather:.85,invert:false},exposure:0,temperature:0,tint:0,contrast:0,saturation:0,opacity:1,enabled:true});
  // Coordinates are clipped to the renderer's normalized gradient contract.
  for(const l of layers)for(const k of ['x1','x2','y1','y2'])l.gradient[k]=clamp(l.gradient[k],0,1);
  for(let pass=0;pass<4;pass++)for(const l of layers){for(const key of ['exposure','temperature','tint','contrast','saturation']){const old=l[key],step=(key==='exposure'?.35:12)*2**(-pass),limit=key==='exposure'?1.5:key==='saturation'?50:30;let best=old;for(const dir of [-1,1]){l[key]=clamp(old+dir*step,-limit,limit);const e=evaluate();if(e<error){error=e;best=l[key];}}l[key]=best;}progress({phase:'局部蒙版拟合',progress:(pass*6+layers.indexOf(l)+1)/24});}
  layers=layers.filter(l=>['exposure','temperature','tint','contrast','saturation'].some(k=>Math.abs(l[k])>1e-6));
 }
 return {params,layers,initial,error};
}
export function fitColorMap(source,target,valid,size=9){
 const n=size**3,values=new Float32Array(n*3),weights=new Float32Array(n);
 for(let i=0;i<source.length;i+=4){if(valid&&!valid[i/4])continue;const p=[0,1,2].map(c=>clamp(source[i+c]/255,0,1)*(size-1)),b=p.map(Math.floor),f=p.map((v,c)=>v-b[c]);for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++){const weight=(x?f[0]:1-f[0])*(y?f[1]:1-f[1])*(z?f[2]:1-f[2]),j=(Math.min(b[2]+z,size-1)*size+Math.min(b[1]+y,size-1))*size+Math.min(b[0]+x,size-1);weights[j]+=weight;for(let c=0;c<3;c++)values[j*3+c]+=(target[i+c]-source[i+c])/255*weight;}}
 for(let i=0;i<n;i++)if(weights[i])for(let c=0;c<3;c++)values[i*3+c]/=weights[i];
 // Smooth and fill unsampled colors, tethered to measured correspondences.
 for(let pass=0;pass<24;pass++){const next=values.slice();for(let z=0;z<size;z++)for(let y=0;y<size;y++)for(let x=0;x<size;x++){const j=(z*size+y)*size+x,neighbors=[];for(const [dx,dy,dz] of [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])if(x+dx>=0&&x+dx<size&&y+dy>=0&&y+dy<size&&z+dz>=0&&z+dz<size)neighbors.push(((z+dz)*size+y+dy)*size+x+dx);const anchor=Math.min(20,weights[j]);for(let c=0;c<3;c++)next[j*3+c]=(values[j*3+c]*anchor+neighbors.reduce((s,k)=>s+values[k*3+c],0))/(anchor+neighbors.length);}values.set(next);}
 return {kind:'lut',size,values,strength:1};
}
export function fitResidual(source,target,w,h,valid){const values=new Float32Array(w*h*3);for(let i=0;i<w*h;i++)if(!valid||valid[i])for(let c=0;c<3;c++)values[i*3+c]=(target[i*4+c]-source[i*4+c])/255;return {kind:'residual',width:w,height:h,values,strength:1};}
export function evaluateMap(source,target,w,h,map,valid){const out=new Uint8ClampedArray(source),rgb=new Float64Array(3);for(let i=0;i<w*h;i++){matchColor(map,...[0,1,2].map(c=>source[i*4+c]/255),(i%w+.5)/w,(Math.floor(i/w)+.5)/h,rgb);for(let c=0;c<3;c++)out[i*4+c]=rgb[c]*255;}return colorError(out,target,valid);}
