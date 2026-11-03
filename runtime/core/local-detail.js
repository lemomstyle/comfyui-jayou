import {dehazeRGB,dehazeSupport} from './dehaze.js';
import {denoiseRGB,denoiseDefaults,denoiseSupport} from './denoise.js';
import {textureRGB,textureSupport} from './texture.js';
import {clarityRGB,claritySupport} from './clarity.js';
import {sharpenRGB,sharpeningDefaults,sharpeningSupport} from './sharpen.js';

// Include the accumulated support of sequential filters. Cropping to the mask
// alone would change neighborhood statistics and leave visible boundary seams.
export function localDetailRegion(weights,w,h,padding=0){
 let left=w,top=h,right=-1,bottom=-1;
 for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(weights[y*w+x]>0){left=Math.min(left,x);right=Math.max(right,x);top=Math.min(top,y);bottom=Math.max(bottom,y);}
 if(right<0)return null;
 const x=Math.max(0,left-padding),y=Math.max(0,top-padding);
 return {x,y,width:Math.min(w,right+padding+1)-x,height:Math.min(h,bottom+padding+1)-y};
}
export function applyMaskedDetail(stage,w,h,layer,weights,scale=1){
 const p={...denoiseDefaults,...sharpeningDefaults,...layer};
 const padding=dehazeSupport(p,scale)+denoiseSupport(p,scale)+textureSupport(p,scale)+claritySupport(p,scale)+sharpeningSupport(p,scale);
 if(!padding)return stage;
 const region=localDetailRegion(weights,w,h,padding);if(!region)return stage;
 const {x,y,width:rw,height:rh}=region,filtered=new Float32Array(rw*rh*3);
 for(let row=0;row<rh;row++){const start=((row+y)*w+x)*3;filtered.set(stage.subarray(start,start+rw*3),row*rw*3);}
 dehazeRGB(filtered,rw,rh,p,scale);denoiseRGB(filtered,rw,rh,p,scale);textureRGB(filtered,rw,rh,p,scale);clarityRGB(filtered,rw,rh,p,scale);sharpenRGB(filtered,rw,rh,p,scale);
 for(let row=0;row<rh;row++)for(let col=0;col<rw;col++){
  const index=(row+y)*w+x+col,a=weights[index];if(!a)continue;
  const j=index*3,k=(row*rw+col)*3;
  for(let c=0;c<3;c++)stage[j+c]+=(filtered[k+c]-stage[j+c])*a;
 }
 return stage;
}
