import {boxMean} from './clarity.js';
// Native band-limited luminance detail. Fine prefilter reduces amplification of
// single-pixel noise, broad local variance suppresses changes on strong edges.
export const textureDefaults={texture:0};
const radii=scale=>[Math.max(1,Math.round(scale)),Math.max(2,Math.round(4*scale))];
export function textureSupport(p,scale=1){return p.texture?radii(scale).reduce((sum,r)=>sum+2*r,0):0;}
const smooth=(a,w,h,r)=>boxMean(boxMean(a,w,h,r),w,h,r);
export function textureRGB(rgb,w,h,p,scale=1){
 if(!p.texture)return rgb;
 const y=new Float32Array(w*h);for(let i=0,j=0;i<y.length;i++,j+=3)y[i]=.2126*rgb[j]+.7152*rgb[j+1]+.0722*rgb[j+2];
 const [fineRadius,broadRadius]=radii(scale),fine=smooth(y,w,h,fineRadius),base=smooth(fine,w,h,broadRadius),square=new Float32Array(y.length);
 for(let i=0;i<y.length;i++)square[i]=fine[i]*fine[i];
 const moment=smooth(square,w,h,broadRadius),strength=p.texture/100;
 for(let i=0,j=0;i<y.length;i++,j+=3){
  const variance=Math.max(0,moment[i]-base[i]*base[i]),edgeProtection=.0025/(.0025+variance);
  const tone=Math.max(0,Math.min(1,y[i])),toneProtection=16*tone*tone*(1-tone)*(1-tone);
  const delta=strength*edgeProtection*toneProtection*(fine[i]-base[i]);
  rgb[j]+=delta;rgb[j+1]+=delta;rgb[j+2]+=delta;
 }
 return rgb;
}
