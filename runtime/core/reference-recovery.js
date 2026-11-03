// Optional reference pixels, separate from the baseline parameter recipe.
const checked=new WeakSet();
const smooth=(a,b,v)=>{const t=Math.max(0,Math.min(1,(v-a)/(b-a)));return t*t*(3-2*t);};
function exposureFailure(r,g,b,tr,tg,tb,highlights=1,shadows=1){
 const y=.2126*r+.7152*g+.0722*b,ty=.2126*tr+.7152*tg+.0722*tb;
 return Math.max(highlights*smooth(.86,.98,Math.max(r,g,b))*smooth(.01,.065,y-ty),shadows*(1-smooth(.015,.12,y))*smooth(.015,.10,ty-y));
}
export function validateRecovery(a){
 if(!a)return null;if(checked.has(a))return a;
 if(!Number.isInteger(a.width)||!Number.isInteger(a.height)||a.width<1||a.height<1||a.width*a.height>100e6||!(a.pixels instanceof Uint8ClampedArray||a.pixels instanceof Uint8Array)||a.pixels.length!==a.width*a.height*4)throw Error('参考细节图像无效');
 if(!Number.isInteger(a.maskWidth)||!Number.isInteger(a.maskHeight)||a.maskWidth<1||a.maskHeight<1||a.maskWidth*a.maskHeight>1048576||!(a.mask instanceof Uint8Array)||a.mask.length!==a.maskWidth*a.maskHeight)throw Error('参考细节区域无效');
 if(!Array.isArray(a.matrix)||a.matrix.length!==9||!a.matrix.every(Number.isFinite)||!Number.isFinite(a.strength)||a.strength<0||a.strength>1)throw Error('参考细节定位无效');
 if(a.transform&&(!Array.isArray(a.transform)||a.transform.length!==6||!a.transform.every(Number.isFinite)))throw Error('参考细节坐标无效');
 for(const key of ['highlights','shadows','detailProtection','contentStrength'])if(a[key]!==undefined&&(!Number.isFinite(a[key])||a[key]<0||a[key]>1))throw Error('参考细节分区强度无效');
 if(a.contentMask){const w=a.contentWidth??a.maskWidth,h=a.contentHeight??a.maskHeight;if(!Number.isInteger(w)||!Number.isInteger(h)||w<1||h<1||w*h>16777216||!(a.contentMask instanceof Uint8Array)||a.contentMask.length!==w*h)throw Error('参考内容区域无效');}
 if(a.contentToneMask&&(!(a.contentToneMask instanceof Uint8Array)||a.contentToneMask.length!==a.contentMask?.length))throw Error('参考颜色过渡区域无效');
 if(a.toneField&&(!(a.toneField instanceof Float32Array)||a.toneField.length!==a.mask.length*3||!a.toneField.every(Number.isFinite)))throw Error('参考过渡场无效');
 checked.add(a);return a;
}
function sample(p,w,h,x,y,c,stride){const xx=Math.max(0,Math.min(w-1,x*w-.5)),yy=Math.max(0,Math.min(h-1,y*h-.5)),ix=Math.floor(xx),iy=Math.floor(yy),fx=xx-ix,fy=yy-iy;return p[(iy*w+ix)*stride+c]*(1-fx)*(1-fy)+p[(iy*w+Math.min(w-1,ix+1))*stride+c]*fx*(1-fy)+p[(Math.min(h-1,iy+1)*w+ix)*stride+c]*(1-fx)*fy+p[(Math.min(h-1,iy+1)*w+Math.min(w-1,ix+1))*stride+c]*fx*fy;}
// Evaluate the original input, before tone curves can clip surviving strokes.
// Two supporting samples suppress isolated noise; the neighborhood also protects
// pale antialiased fringes around dark lines. No full-resolution mask allocation.
export function sourceDetailProtection(source,index){
 const {data,w,h,inputScale}=source,x=index%w,y=Math.floor(index/w);
 let low=Infinity,secondLow=Infinity,high=-Infinity,secondHigh=-Infinity;
 for(let yy=Math.max(0,y-2);yy<=Math.min(h-1,y+2);yy++)for(let xx=Math.max(0,x-2);xx<=Math.min(w-1,x+2);xx++){
  const i=(yy*w+xx)*4,v=(.2126*data[i]+.7152*data[i+1]+.0722*data[i+2])/inputScale;
  if(v<low){secondLow=low;low=v;}else if(v<secondLow)secondLow=v;
  if(v>high){secondHigh=high;high=v;}else if(v>secondHigh)secondHigh=v;
 }
 const i=index*4,center=(.2126*data[i]+.7152*data[i+1]+.0722*data[i+2])/inputScale;
 return Number.isFinite(secondLow)&&Number.isFinite(secondHigh)?smooth(.025,.10,secondHigh-secondLow)*smooth(.004,.025,secondHigh-center):0;
}
const detailDirections=[[1,0],[0,1],[1,1],[1,-1]];
export function reliableSourceDetail(source,index){
 const base=sourceDetailProtection(source,index);if(!base)return 0;
 const {data,w,h,inputScale:s}=source,x=index%w,y=Math.floor(index/w),l=(xx,yy)=>{const i=(yy*w+xx)*4;return (.2126*data[i]+.7152*data[i+1]+.0722*data[i+2])/s;},center=l(x,y);let continuity=0;
 for(const [dx,dy] of detailDirections)if(x-dx>=0&&x+dx<w&&y-dy>=0&&y-dy<h&&y+dy>=0&&y+dy<h){const a=l(x-dx,y-dy),b=l(x+dx,y+dy);continuity=Math.max(continuity,1-smooth(.015,.07,Math.max(Math.abs(a-center),Math.abs(b-center))));}
 return base*continuity*(1-smooth(.94,.995,center));
}
export function recoverReference(a,x,y,r,g,b,out,source,index){
 out[0]=r;out[1]=g;out[2]=b;if(!a||!a.strength)return out;
 if(a.transform){const m=a.transform,xx=m[0]*x+m[1]*y+m[2];y=m[3]*x+m[4]*y+m[5];x=xx;}
 if(x<0||x>1||y<0||y>1)return out;
 let alpha=sample(a.mask,a.maskWidth,a.maskHeight,x,y,0,1)/255*a.strength;
 const content=a.contentMask?(a.contentStrength||0)*a.strength*sample(a.contentMask,a.contentWidth??a.maskWidth,a.contentHeight??a.maskHeight,x,y,0,1)/255:0;const contentTone=a.contentToneMask?(a.contentStrength||0)*a.strength*sample(a.contentToneMask,a.contentWidth??a.maskWidth,a.contentHeight??a.maskHeight,x,y,0,1)/255:content;if(alpha<=0&&content<=0&&contentTone<=0)return out;
 const m=a.matrix,d=m[6]*x+m[7]*y+m[8];if(Math.abs(d)<1e-8)return out;
 const u=(m[0]*x+m[1]*y+m[2])/d,v=(m[3]*x+m[4]*y+m[5])/d;if(u<0||u>1||v<0||v>1)return out;
 const tr=sample(a.pixels,a.width,a.height,u,v,0,4)/255,tg=sample(a.pixels,a.width,a.height,u,v,1,4)/255,tb=sample(a.pixels,a.width,a.height,u,v,2,4)/255;
 // New recovery separates smooth tone correction from reference texture.
 if(a.adaptive===2){
  let gate=exposureFailure(r,g,b,tr,tg,tb,a.highlights??1,a.shadows??0);
  if(source){const i=index*4,s=source.inputScale;gate=Math.max(gate,exposureFailure(source.data[i]/s,source.data[i+1]/s,source.data[i+2]/s,tr,tg,tb,a.highlights??1,a.shadows??0));}
  alpha*=gate;
  const protection=alpha>0&&source&&a.detailProtection?reliableSourceDetail(source,index)*a.detailProtection:0;
  for(let c=0;c<3;c++){
   const delta=a.toneField?sample(a.toneField,a.maskWidth,a.maskHeight,x,y,c,3):0,ref=c===0?tr:c===1?tg:tb;
   // Protection limits imported texture only; tone/color changes remain active.
   out[c]+=alpha*(delta+(1-protection)*(ref-out[c]-delta));
   if(a.contentToneMask)out[c]+=delta*contentTone+(ref-out[c]-delta)*content;else out[c]+=(ref-out[c])*content;
  }
 }else{
  if(a.adaptive===1||a.highlights!==undefined||a.shadows!==undefined)alpha*=exposureFailure(r,g,b,tr,tg,tb,a.highlights??1,a.shadows??1);
  if(alpha>0&&a.detailProtection&&source)alpha*=1-a.detailProtection*sourceDetailProtection(source,index);
  out[0]+=(tr-r)*alpha;out[1]+=(tg-g)*alpha;out[2]+=(tb-b)*alpha;
 }
 return out;
}
export function transformRecovery(a,inverse){if(!a)return a;const m=a.transform||[1,0,0,0,1,0],convert=p=>({x:m[0]*p.x+m[1]*p.y+m[2],y:m[3]*p.x+m[4]*p.y+m[5]}),o=convert(inverse({x:0,y:0})),x=convert(inverse({x:1,y:0})),y=convert(inverse({x:0,y:1}));return {...a,transform:[x.x-o.x,y.x-o.x,o.x,x.y-o.y,y.y-o.y,o.y]};}
export function recoveryMask(rendered,target,w,h,valid,confidence,{adaptive=false,source}={}){
 if(adaptive){
  const seeds=new Uint8Array(w*h),out=new Uint8Array(w*h);
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
   const k=y*w+x,i=k*4;if(!valid[k])continue;
   const r=rendered[i],g=rendered[i+1],b=rendered[i+2],s=source instanceof Float32Array?1:source instanceof Uint16Array?65535:255,sr=source?source[i]/s:r,sg=source?source[i+1]/s:g,sb=source?source[i+2]/s:b;
   let supported=confidence[k]>=.5;
   if(source&&Math.min(sr,sg,sb)>.985&&target[i]+target[i+1]+target[i+2]<735)supported=true;
   // Flat clipped pixels lack gradient correspondence. Require nearby reliable
   // alignment before admitting them; never admit invalid warped borders.
   if(!supported&&(Math.min(sr,sg,sb)>.94||Math.max(sr,sg,sb)<.035)){
    let reliable=0,total=0;
    for(let yy=Math.max(0,y-3);yy<=Math.min(h-1,y+3);yy++)for(let xx=Math.max(0,x-3);xx<=Math.min(w-1,x+3);xx++){const j=yy*w+xx;if(valid[j]){total++;if(confidence[j]>=.5)reliable++;}}
    supported=reliable>=Math.max(3,total*.25);
   }
   if(supported&&Math.max(exposureFailure(r,g,b,target[i]/255,target[i+1]/255,target[i+2]/255),exposureFailure(sr,sg,sb,target[i]/255,target[i+1]/255,target[i+2]/255))>.03)seeds[k]=255;
  }
  // A small proposal margin catches fine highlights lost by analysis resizing.
  // Native-resolution exposure tests, not this dilation, determine final alpha.
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){const k=y*w+x;if(!valid[k])continue;let value=seeds[k];for(let yy=Math.max(0,y-2);yy<=Math.min(h-1,y+2);yy++)for(let xx=Math.max(0,x-2);xx<=Math.min(w-1,x+2);xx++){if(seeds[yy*w+xx])value=Math.max(value,Math.round(255*(1-Math.max(Math.abs(xx-x),Math.abs(yy-y))/3)));}out[k]=value;}
  return out;
 }
 const mask=new Uint8Array(w*h),smooth=(a,b,v)=>{const t=Math.max(0,Math.min(1,(v-a)/(b-a)));return t*t*(3-2*t);};
 for(let k=0;k<mask.length;k++){
  if(!valid[k]||confidence[k]<.5)continue;
  const i=k*4,r=rendered[i],g=rendered[i+1],b=rendered[i+2],t=[target[i]/255,target[i+1]/255,target[i+2]/255],y=.2126*r+.7152*g+.0722*b,ty=.2126*t[0]+.7152*t[1]+.0722*t[2];
  const hi=smooth(.88,.99,Math.max(r,g,b))*smooth(.015,.09,y-ty),lo=(1-smooth(.015,.12,y))*smooth(.015,.10,ty-y);
  mask[k]=Math.round(255*Math.max(hi,lo));
 }
 // Feather only inside valid matching regions; never leak into unmatched edges.
 const out=new Uint8Array(mask.length);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){const k=y*w+x;if(!valid[k]||confidence[k]<.5)continue;let sum=0,n=0;for(let yy=Math.max(0,y-2);yy<=Math.min(h-1,y+2);yy++)for(let xx=Math.max(0,x-2);xx<=Math.min(w-1,x+2);xx++){sum+=mask[yy*w+xx];n++;}out[k]=Math.min(mask[k],sum/n);}
 return out;
}

// Reuse immutable pixel/mask assets per render worker instead of cloning them
// for every slider event and every export strip.
const assetKeys=['pixels','mask','toneField','contentMask','contentToneMask'];
export function packRecovery(message,cache){
 const a=message.params?.referenceRecovery;
 if(!a){if(cache.pixels){for(const key of assetKeys)delete cache[key];return {...message,clearRecovery:true};}return message;}
 const changed=assetKeys.some(key=>cache[key]!==a[key]);if(changed){for(const key of assetKeys)cache[key]=a[key];cache.key=(cache.key||0)+1;}
 const recipe={...a},asset={key:cache.key};for(const key of assetKeys){delete recipe[key];if(changed)asset[key]=a[key];}
 return {...message,params:{...message.params,referenceRecovery:recipe},recoveryAsset:asset};
}
export function unpackRecovery(message,cache){
 if(message.clearRecovery){for(const key of assetKeys)delete cache[key];delete cache.key;}
 const asset=message.recoveryAsset;if(!asset)return message;
 if(asset.pixels){for(const key of assetKeys)cache[key]=asset[key];cache.key=asset.key;}
 if(cache.key!==asset.key)throw Error('参考细节缓存失效');
 const a={...message.params.referenceRecovery};for(const key of assetKeys)if(cache[key]!==undefined)a[key]=cache[key];
 return {...message,params:{...message.params,referenceRecovery:a}};
}
// Output support of the actual tone/content contribution, not a new selection.
export function effectiveCoverage(a,x,y,r,g,b,source,index){
 if(!a||!a.strength)return 0;const m=a.matrix,d=m[6]*x+m[7]*y+m[8];if(Math.abs(d)<1e-8)return 0;
 const u=(m[0]*x+m[1]*y+m[2])/d,v=(m[3]*x+m[4]*y+m[5])/d;if(u<0||u>1||v<0||v>1)return 0;
 const tr=sample(a.pixels,a.width,a.height,u,v,0,4)/255,tg=sample(a.pixels,a.width,a.height,u,v,1,4)/255,tb=sample(a.pixels,a.width,a.height,u,v,2,4)/255;
 let gate=exposureFailure(r,g,b,tr,tg,tb,a.highlights??1,a.shadows??0);
 if(source){const i=index*4;gate=Math.max(gate,exposureFailure(source.data[i],source.data[i+1],source.data[i+2],tr,tg,tb,a.highlights??1,a.shadows??0));}
 const alpha=sample(a.mask,a.maskWidth,a.maskHeight,x,y,0,1)/255*gate,content=a.contentMask?sample(a.contentMask,a.contentWidth,a.contentHeight,x,y,0,1)/255*(a.contentStrength||0):0,tone=a.contentToneMask?sample(a.contentToneMask,a.contentWidth,a.contentHeight,x,y,0,1)/255*(a.contentStrength||0):0;
 return Math.max(alpha,content,tone)*a.strength;
}
