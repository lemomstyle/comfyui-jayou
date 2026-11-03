import {dehazeRGB,dehazeSupport} from './dehaze.js';
import {textureSupport} from './texture.js';
import {claritySupport} from './clarity.js';
import {denoiseSupport} from './denoise.js';
// Native luminance unsharp mask. Radius is measured in original-image pixels.
// Operates on float RGB without clipping extended colors; grain is applied later.
export const sharpeningDefaults={sharpenAmount:0,sharpenRadius:1,sharpenDetail:25,sharpenMasking:0};
export function sharpeningSupport(p,scale=1){return p.sharpenAmount>0?Math.ceil(3*Math.max(.15,(p.sharpenRadius??1)*scale)):0;}
function gaussian(src,w,h,sigma){
 const radius=Math.ceil(3*sigma),kernel=new Float64Array(2*radius+1);let sum=0;
 for(let k=-radius;k<=radius;k++){kernel[k+radius]=Math.exp(-k*k/(2*sigma*sigma));sum+=kernel[k+radius];}for(let i=0;i<kernel.length;i++)kernel[i]/=sum;
 const tmp=new Float32Array(src.length),out=new Float32Array(src.length);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=0;for(let k=-radius;k<=radius;k++)v+=src[y*w+Math.max(0,Math.min(w-1,x+k))]*kernel[k+radius];tmp[y*w+x]=v;}
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=0;for(let k=-radius;k<=radius;k++)v+=tmp[Math.max(0,Math.min(h-1,y+k))*w+x]*kernel[k+radius];out[y*w+x]=v;}
 return out;
}
export function sharpenRGB(rgb,w,h,p,scale=1){
 if(!p.sharpenAmount)return rgb;
 const y=new Float32Array(w*h);for(let i=0,j=0;i<y.length;i++,j+=3)y[i]=.2126*rgb[j]+.7152*rgb[j+1]+.0722*rgb[j+2];
 const broad=gaussian(y,w,h,Math.max(.15,p.sharpenRadius*scale)),fine=gaussian(y,w,h,Math.max(.15,Math.min(.5,p.sharpenRadius)*scale));
 const detail=p.sharpenDetail/100,masking=p.sharpenMasking/100,amount=p.sharpenAmount/100;
 for(let i=0,j=0;i<y.length;i++,j+=3){const high=y[i]-broad[i],micro=y[i]-fine[i],threshold=.06*masking;
 // Smooth threshold suppresses low-contrast texture/noise as masking increases.
 const t=masking?Math.max(0,Math.min(1,(Math.abs(high)-threshold*.15)/Math.max(.00001,threshold))):1,weight=t*t*(3-2*t);
 const delta=amount*weight*((1-detail)*high+detail*micro*2);
 rgb[j]+=delta;rgb[j+1]+=delta;rgb[j+2]+=delta;
 }
 return rgb;
}
export function renderSupport(p,width,height,scale=1,layers=[]){const local=layers.filter(l=>l.enabled!==false&&!l.protect&&(l.opacity??1)!==0).reduce((n,l)=>n+dehazeSupport(l,scale)+denoiseSupport(l,scale)+textureSupport(l,scale)+claritySupport(l,scale)+sharpeningSupport(l,scale),0);return (p.referenceRecovery?.detailProtection?2:0)+local+dehazeSupport(p,scale)+textureSupport(p,scale)+claritySupport(p,scale)+denoiseSupport(p,scale)+sharpeningSupport(p,scale)+(p.highlightSoftness?Math.max(1,Math.round(Math.max(width,height)/500)):0);}
