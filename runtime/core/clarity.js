// Edge-aware, medium-scale luminance contrast. Native units, not Adobe's algorithm.
// Two box passes form a guided-filter base; each pass needs radius pixels of halo.
export const clarityDefaults={clarity:0};
const radiusAt=scale=>Math.max(1,Math.round(12*scale));
export function claritySupport(p,scale=1){return p.clarity?2*radiusAt(scale):0;}
export function boxMean(src,w,h,r){
 const tmp=new Float32Array(src.length),out=new Float32Array(src.length),n=2*r+1;
 for(let y=0;y<h;y++){const row=y*w;let sum=0;for(let k=-r;k<=r;k++)sum+=src[row+Math.max(0,Math.min(w-1,k))];for(let x=0;x<w;x++){tmp[row+x]=sum/n;sum+=src[row+Math.min(w-1,x+r+1)]-src[row+Math.max(0,x-r)];}}
 for(let x=0;x<w;x++){let sum=0;for(let k=-r;k<=r;k++)sum+=tmp[Math.max(0,Math.min(h-1,k))*w+x];for(let y=0;y<h;y++){out[y*w+x]=sum/n;sum+=tmp[Math.min(h-1,y+r+1)*w+x]-tmp[Math.max(0,y-r)*w+x];}}
 return out;
}
export function clarityRGB(rgb,w,h,p,scale=1){
 if(!p.clarity)return rgb;
 const y=new Float32Array(w*h),square=new Float32Array(w*h);
 for(let i=0,j=0;i<y.length;i++,j+=3){y[i]=.2126*rgb[j]+.7152*rgb[j+1]+.0722*rgb[j+2];square[i]=y[i]*y[i];}
 const radius=radiusAt(scale),mean=boxMean(y,w,h,radius),moment=boxMean(square,w,h,radius);
 const a=new Float32Array(y.length),b=new Float32Array(y.length);
 for(let i=0;i<y.length;i++){const variance=Math.max(0,moment[i]-mean[i]*mean[i]);a[i]=variance/(variance+.0025);b[i]=mean[i]*(1-a[i]);}
 const ma=boxMean(a,w,h,radius),mb=boxMean(b,w,h,radius),strength=p.clarity/100*2;
 for(let i=0,j=0;i<y.length;i++,j+=3){
  const tone=Math.max(0,Math.min(1,y[i])),protect=16*tone*tone*(1-tone)*(1-tone);
  const detail=y[i]-(ma[i]*y[i]+mb[i]);
  const delta=strength*protect*detail;
  rgb[j]+=delta;rgb[j+1]+=delta;rgb[j+2]+=delta;
 }
 return rgb;
}
