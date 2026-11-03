import {createPixelCache} from './pixel-cache.js';
import {boxMean} from './clarity.js';
import {encodedToLinear,linearToEncoded} from './linear-color.js';
// Native dark-channel transmission with guided refinement. Neutral airlight is
// fixed at scene-linear white, so tiles/preview crops never estimate different
// atmospheric colors. This is not Adobe's proprietary Dehaze algorithm.
const cache=createPixelCache(64*1024*1024,2);
export const clearDehazeCache=()=>cache.clear();
export const dehazeCacheInfo=()=>cache.info();
export const dehazeDefaults={dehaze:0};
const radiusAt=scale=>Math.max(1,Math.round(16*scale));
export const dehazeSupport=(p,scale=1)=>p.dehaze?3*radiusAt(scale):0;
export function minimumFilter(src,w,h,r){
 const tmp=new Float32Array(src.length),out=new Float32Array(src.length),q=new Int32Array(Math.max(w,h));
 function line(n,read,write){let head=0,tail=0,next=0;for(let x=0;x<n;x++){
  const end=Math.min(n-1,x+r);while(next<=end){while(tail>head&&read(q[tail-1])>=read(next))tail--;q[tail++]=next++;}
  while(q[head]<x-r)head++;write(x,read(q[head]));
 }}
 for(let y=0;y<h;y++)line(w,x=>src[y*w+x],(x,v)=>tmp[y*w+x]=v);
 for(let x=0;x<w;x++)line(h,y=>tmp[y*w+x],(y,v)=>out[y*w+x]=v);
 return out;
}
function prepareDehaze(rgb,w,h,radius){
 const key=`${w}:${h}:${radius}`,cached=cache.get(rgb,key);if(cached)return cached;
 const n=w*h,linear=new Float64Array(rgb.length),guide=new Float32Array(n),dark=new Float32Array(n),square=new Float32Array(n);
 for(let i=0,j=0;i<n;i++,j+=3){const r=linear[j]=encodedToLinear(rgb[j]),g=linear[j+1]=encodedToLinear(rgb[j+1]),b=linear[j+2]=encodedToLinear(rgb[j+2]);guide[i]=Math.max(0,Math.min(1,.2126*r+.7152*g+.0722*b));square[i]=guide[i]*guide[i];dark[i]=Math.max(0,Math.min(1,r,g,b));}
 const minimum=minimumFilter(dark,w,h,radius),cross=new Float32Array(n);
 for(let i=0;i<n;i++)cross[i]=minimum[i]*guide[i];
 const mi=boxMean(guide,w,h,radius),mp=boxMean(minimum,w,h,radius),mii=boxMean(square,w,h,radius),mip=boxMean(cross,w,h,radius);
 for(let i=0;i<n;i++){const a=(mip[i]-mi[i]*mp[i])/(Math.max(0,mii[i]-mi[i]*mi[i])+.001);square[i]=a;cross[i]=mp[i]-a*mi[i];}
 const ma=boxMean(square,w,h,radius),mb=boxMean(cross,w,h,radius);
 // Guided filtering is linear in its signal for a fixed guide. Refine the
 // dark channel once; each slider position derives t = 1 - 0.8 * amount * q.
 for(let i=0;i<n;i++)minimum[i]=ma[i]*guide[i]+mb[i];
 const value={linear,dark:minimum};cache.put(rgb,key,value,linear.byteLength+minimum.byteLength);return value;
}
export function dehazeRGB(rgb,w,h,p,scale=1){
 if(!p.dehaze)return rgb;
 const strength=p.dehaze/100;
 if(strength<0){const t=1+strength*.6;for(let i=0;i<rgb.length;i++)rgb[i]=linearToEncoded(encodedToLinear(rgb[i])*t+1-t);return rgb;}
 const prepared=prepareDehaze(rgb,w,h,radiusAt(scale));
 for(let i=0,j=0;i<w*h;i++,j+=3){const t=Math.max(.2,Math.min(1,1-.8*strength*prepared.dark[i]));for(let c=0;c<3;c++)rgb[j+c]=linearToEncoded((prepared.linear[j+c]-1)/t+1);}
 return rgb;
}
