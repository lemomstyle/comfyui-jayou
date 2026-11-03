import {validateLocalLight,applyLocalLight} from './local-light.js';
import {validateColorRefinement,refineColor} from './color-refinement.js';
import {validateRecovery,recoverReference} from './reference-recovery.js';
import {validateMatchMap,matchColor} from './same-image-map.js';
import {calibrationDefaults,calibrationMatrix,shadowCalibration} from './calibration.js';
import {gradingWeights} from './grading-balance.js';
import {blackWhiteValue,validateBlackWhite} from './black-white.js';
import {maskWeightCache} from './mask-weight-cache.js';
import {dehazeRGB,dehazeDefaults} from './dehaze.js';
import {colorRangeWeight,validateColorRange} from './color-range.js';
import {applyMaskedDetail} from './local-detail.js';
import {finalMaskWeight} from './mask-operations.js';
import {brushBounds,composeBrushWeight} from './brush-composite.js';
import {luminanceRangeWeight} from './luminance-range.js';
import {textureRGB,textureDefaults} from './texture.js';
import {clarityRGB,clarityDefaults} from './clarity.js';
import {denoiseRGB,denoiseDefaults} from './denoise.js';
import {sharpenRGB,sharpeningDefaults} from './sharpen.js';
import {defaultPointCurves,curveChannels,validatePointCurve,compilePointCurve} from './point-curves.js';
import {validLut,getLut,sampleLut,sampleLutExtended} from './lut.js';
import {geometryCoverage} from './gradients.js';
// Sizhui SDR renderer v1. All controls are native to this engine, not Lightroom units.
export const VERSION='sizhui-srgb-v2';
export const bands=['red','orange','yellow','green','aqua','blue','purple','magenta'];
export const centers=[0,30,60,120,180,240,275,315];
export const ranges={...Object.fromEntries(Object.keys(calibrationDefaults).map(k=>[k,[-100,100]])),gradingGlobalHue:[0,360],gradingGlobalSaturation:[0,40],gradingGlobalLuminance:[-100,100],gradingShadowLuminance:[-100,100],gradingMidtoneLuminance:[-100,100],gradingHighlightLuminance:[-100,100],gradingBalance:[-100,100],gradingBlending:[0,100],dehaze:[-100,100],texture:[-100,100],clarity:[-100,100],noiseLuminance:[0,100],noiseLuminanceDetail:[0,100],noiseColor:[0,100],noiseColorDetail:[0,100],sharpenAmount:[0,150],sharpenRadius:[.5,3],sharpenDetail:[0,100],sharpenMasking:[0,100],exposure:[-3,3],contrast:[-60,60],temperature:[-50,50],tint:[-50,50],whites:[-100,100],blacks:[-100,100],vibrance:[-100,100],shadows:[-60,60],highlights:[-60,60],saturation:[-70,70],highlightSoftness:[0,70],grain:[0,40],vignette:[0,60]};
export const clamp=(v,a=0,b=1)=>Math.max(a,Math.min(b,v));
export function defaults(){return {...calibrationDefaults,gradingGlobalHue:0,gradingGlobalSaturation:0,gradingGlobalLuminance:0,gradingShadowLuminance:0,gradingMidtoneLuminance:0,gradingHighlightLuminance:0,gradingBalance:0,gradingBlending:50,blackWhite:false,blackWhiteMix:Array(8).fill(0),...dehazeDefaults,...textureDefaults,...clarityDefaults,...denoiseDefaults,...sharpeningDefaults,exposure:0,contrast:0,temperature:0,tint:0,whites:0,blacks:0,vibrance:0,shadows:0,highlights:0,saturation:0,highlightSoftness:0,grain:0,vignette:0,pointCurves:defaultPointCurves(),curve:[0,.25,.5,.75,1],hsl:bands.map(()=>({hue:0,saturation:0,lightness:0})),grading:{shadows:{hue:220,saturation:0},midtones:{hue:40,saturation:0},highlights:{hue:50,saturation:0}}};}
export function validateParams(p){
 if(!p || typeof p!=='object')throw Error('参数必须为对象');
 const q={...defaults(),...validateBlackWhite(p)};
 for(const [k,[a,b]] of Object.entries(ranges)){if(p[k]===undefined&&[...Object.keys(calibrationDefaults),'gradingGlobalHue','gradingGlobalSaturation','gradingGlobalLuminance','gradingShadowLuminance','gradingMidtoneLuminance','gradingHighlightLuminance','gradingBalance','gradingBlending','dehaze','whites','blacks','vibrance',...Object.keys(textureDefaults),...Object.keys(clarityDefaults),...Object.keys(sharpeningDefaults),...Object.keys(denoiseDefaults)].includes(k))continue;if(!Number.isFinite(p[k])||p[k]<a||p[k]>b)throw Error(`参数 ${k} 超出范围 [${a}, ${b}]`);q[k]=p[k];}
 if(!Array.isArray(p.curve)||p.curve.length!==5||p.curve.some((x,i)=>!Number.isFinite(x)||x<0||x>1||(i&&x<p.curve[i-1])))throw Error('曲线须有 5 个递增的 0–1 锚点');
 q.curve=[...p.curve];
 if(p.pointCurves!==undefined){for(const channel of curveChannels)q.pointCurves[channel]=validatePointCurve(p.pointCurves[channel]);}
 if(!Array.isArray(p.hsl)||p.hsl.length!==8)throw Error('HSL 须包含 8 个色域');
 q.hsl=p.hsl.map(x=>{for(const k of ['hue','saturation','lightness'])if(!Number.isFinite(x[k])||Math.abs(x[k])>(k==='hue'?40:70))throw Error('HSL 参数不合法');return {hue:x.hue,saturation:x.saturation,lightness:x.lightness};});
 for(const k of Object.keys(q.grading)){const x=p.grading?.[k];if(!x||!Number.isFinite(x.hue)||x.hue<0||x.hue>360||!Number.isFinite(x.saturation)||x.saturation<0||x.saturation>40)throw Error('颜色分级参数不合法');q.grading[k]={hue:x.hue,saturation:x.saturation};}
 if(p.lut){if(!validLut(p.lut))throw Error('未知的 LUT 风格');q.lut=p.lut;}
 if(p.localLight)q.localLight=validateLocalLight(p.localLight);
 if(p.colorRefinement)q.colorRefinement=validateColorRefinement(p.colorRefinement);
 if(p.matchMap)q.matchMap=validateMatchMap(p.matchMap);
 if(p.referenceRecovery)q.referenceRecovery=validateRecovery(p.referenceRecovery);
 return q;
}
export function rgbToHsl(r,g,b){const max=Math.max(r,g,b),min=Math.min(r,g,b),d=max-min,l=(max+min)/2;let h=0,s=0;if(d){s=d/(1-Math.abs(2*l-1));h=max===r?((g-b)/d+6)%6:max===g?(b-r)/d+2:(r-g)/d+4;h*=60;}return [h,s,l];}
export function hslToRgb(h,s,l){h=((h%360)+360)%360;const c=(1-Math.abs(2*l-1))*s,x=c*(1-Math.abs((h/60)%2-1)),m=l-c/2;const a=h<60?[c,x,0]:h<120?[x,c,0]:h<180?[0,c,x]:h<240?[0,x,c]:h<300?[x,0,c]:[c,0,x];return a.map(v=>v+m);}
const lum=(r,g,b)=>.2126*r+.7152*g+.0722*b;
const linear=v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4;
const srgb=v=>v<=.0031308?12.92*v:1.055*v**(1/2.4)-.055;
export function curveValue(v,points){const x=clamp(v)*4,i=Math.min(3,Math.floor(x));return points[i]+(points[i+1]-points[i])*(x-i);}
function smooth(a,b,x){const t=clamp((x-a)/(b-a));return t*t*(3-2*t);}
export function maskWeight(type,r,g,b){const y=lum(r,g,b);return type==='highlights'?smooth(.45,.85,y):type==='shadows'?1-smooth(.15,.55,y):type==='midtones'?Math.max(0,1-Math.abs(y-.5)*2):1;}
// Endpoint controls are smooth luminance-weighted offsets, distinct from broad shadows/highlights.
function endpointDelta(y,whites=0,blacks=0){return whites/400*smooth(.5,1,y)+blacks/400*(1-smooth(0,.5,y));}
// Selectively change weak chroma. Warm hues receive less positive enhancement;
// this is a color heuristic, not semantic skin detection. Preserve luminance and HDR headroom.
function vibranceColor(r,g,b,value){
 const max=Math.max(r,g,b),min=Math.min(r,g,b),chroma=max-min;
 if(chroma<1e-10)return [r,g,b];
 const saturation=clamp(chroma/Math.max(1e-10,max));
 const [h]=rgbToHsl(clamp(r/Math.max(1,max)),clamp(g/Math.max(1,max)),clamp(b/Math.max(1,max)));
 const warm=1-smooth(20,55,Math.abs(h-30));
 const factor=1+value/100*(1-saturation)**2*(value>0?1-.5*warm:1),y=lum(r,g,b);
 return [y+(r-y)*factor,y+(g-y)*factor,y+(b-y)*factor];
}
// Separable box blur: radius scales with image dimensions; no geometry synthesis.
export function blurRGB(src,w,h,r){const tmp=new Float32Array(src.length),out=new Float32Array(src.length);for(let y=0;y<h;y++)for(let c=0;c<3;c++){let sum=0;for(let k=-r;k<=r;k++)sum+=src[(y*w+clamp(k,0,w-1))*3+c];for(let x=0;x<w;x++){tmp[(y*w+x)*3+c]=sum/(2*r+1);sum+=src[(y*w+clamp(x+r+1,0,w-1))*3+c]-src[(y*w+clamp(x-r,0,w-1))*3+c];}}
 for(let x=0;x<w;x++)for(let c=0;c<3;c++){let sum=0;for(let k=-r;k<=r;k++)sum+=tmp[(clamp(k,0,h-1)*w+x)*3+c];for(let y=0;y<h;y++){out[(y*w+x)*3+c]=sum/(2*r+1);sum+=tmp[(clamp(y+r+1,0,h-1)*w+x)*3+c]-tmp[(clamp(y-r,0,h-1)*w+x)*3+c];}}return out;}
function extendedCurve(v,points){if(v<0)return points[0]+v*(points[1]-points[0])*4;if(v>1)return points[4]+(v-1)*(points[4]-points[3])*4;return curveValue(v,points);}
function activePointCurves(curves){return curves&&curveChannels.some(c=>curves[c]?.some(p=>p.x!==p.y));}
function encodeOutput(v,scale){return scale===1?v:scale===65535?Math.round(clamp(v)*65535):clamp(v)*255;}
export function render(data,w,h,p,options={}){
 const highPrecision=data instanceof Uint16Array||data instanceof Float32Array;const inputScale=data instanceof Uint16Array?65535:data instanceof Float32Array?1:255;const outputScale=options.outputFormat==='float32'?1:options.outputFormat==='uint16'?65535:255;
 p=validateParams(p);
 if(p.localLight?.strength){
  const plain={...p};delete plain.localLight;delete plain.referenceRecovery;
  const base=render(data,w,h,plain,{...options,amount:1,outputFormat:'float32'}),out=outputScale===1?base:outputScale===65535?new Uint16Array(data.length):new Uint8ClampedArray(data.length),rgb=new Float64Array(3),amount=clamp(options.amount??1),fw=options.spatial?.width||w,fh=options.spatial?.height||h,ox=options.spatial?.x||0,oy=options.spatial?.y||0,source={data,w,h,inputScale};
  for(let k=0;k<w*h;k++){const i=k*4,x=(k%w+ox+.5)/fw,y=(Math.floor(k/w)+oy+.5)/fh;applyLocalLight(p.localLight,base[i],base[i+1],base[i+2],x,y,rgb);
   if(options.preRecoveryOutput){for(let c=0;c<3;c++)options.preRecoveryOutput[i+c]=rgb[c];options.preRecoveryOutput[i+3]=data[i+3]/inputScale;}
   if(p.referenceRecovery)recoverReference(p.referenceRecovery,x,y,rgb[0],rgb[1],rgb[2],rgb,source,k);
   for(let c=0;c<3;c++)out[i+c]=encodeOutput(data[i+c]/inputScale+(rgb[c]-data[i+c]/inputScale)*amount,outputScale);out[i+3]=data[i+3]/inputScale*outputScale;
  }return out;
 }
 const amount=clamp(options.amount??1),out=options.outputFormat==='float32'?new Float32Array(data.length):options.outputFormat==='uint16'?new Uint16Array(data.length):new Uint8ClampedArray(data.length),stage=new Float32Array(w*h*3),layers=(options.layers||[]).filter(l=>l.enabled!==false&&(l.opacity??1)!==0&&(l.protect||l.dehaze||l.texture||l.clarity||l.noiseLuminance||l.noiseColor||l.sharpenAmount||l.contrast||l.exposure||l.temperature||l.tint||l.shadows||l.highlights||l.whites||l.blacks||l.vibrance||l.saturation||activePointCurves(l.pointCurves)));
 const localCurves=new Map(layers.filter(l=>activePointCurves(l.pointCurves)&&!l.protect).map(l=>[l,Object.fromEntries(curveChannels.map(c=>[c,compilePointCurve(l.pointCurves[c])]))]));
 const pointCurves=Object.fromEntries(curveChannels.map(c=>[c,compilePointCurve(p.pointCurves[c])])),usePointCurves=curveChannels.some(c=>p.pointCurves[c].some(v=>v.x!==v.y));
 const matchPixel=new Float64Array(3),recoverySource={data,w,h,inputScale};
 const lut=getLut(p.lut),lutSampler=highPrecision?sampleLutExtended:sampleLut,lutPixel=new Float64Array(3);
 const gradeWeights=new Float64Array(3),globalGrade=hslToRgb(p.gradingGlobalHue,1,.5).map(c=>(c-.5)*p.gradingGlobalSaturation/100),globalGradeFactor=2**(p.gradingGlobalLuminance/100);
 const grade=Object.fromEntries(Object.entries(p.grading).map(([k,v])=>[k,hslToRgb(v.hue,1,.5).map(c=>(c-.5)*v.saturation/100)]));
 const calibration=calibrationMatrix(p);
 const exp=2**p.exposure,exposureLUT=Array.from({length:256},(_,i)=>srgb(linear(i/255)*exp));
 const useHsl=p.saturation!==0||p.hsl.some(v=>v.hue||v.saturation||v.lightness),useGrade=Object.values(p.grading).some(v=>v.saturation)||p.gradingShadowLuminance||p.gradingMidtoneLuminance||p.gradingHighlightLuminance;
 const fw=options.spatial?.width||w,fh=options.spatial?.height||h,ox=options.spatial?.x||0,oy=options.spatial?.y||0;
 const postBrushes=new Map(),brushes=new Map();
 const prepareBrushes=layer=>{if(!brushes.has(layer)){brushes.set(layer,brushBounds(layer.brushOps,w,h,options.spatial));postBrushes.set(layer,brushBounds(layer.postBrushOps,w,h,options.spatial));}};
 const calculateAlpha=(layer,index)=>{prepareBrushes(layer);const i=index*4;const m=geometryCoverage(layer,index,((index%w)+ox+.5)/fw,(Math.floor(index/w)+oy+.5)/fh,()=>maskWeight(layer.type,data[i]/inputScale,data[i+1]/inputScale,data[i+2]/inputScale));const range=layer.range&&layer.range!=='all'?maskWeight(layer.range,data[i]/inputScale,data[i+1]/inputScale,data[i+2]/inputScale):1;return clamp(finalMaskWeight(composeBrushWeight(m,brushes.get(layer),index)*range*luminanceRangeWeight(layer.luminanceRange,data[i]/inputScale,data[i+1]/inputScale,data[i+2]/inputScale)*colorRangeWeight(layer.colorRange,data[i]/inputScale,data[i+1]/inputScale,data[i+2]/inputScale),layer.inverted,postBrushes.get(layer),index)*(layer.opacity??1));};
 // Bound live per-render weights as well as persistent worker cache memory.
 const cachedWeights=new Map();let weightBudget=32*1024*1024;
 if(options.cacheMaskWeights!==false)for(const layer of layers){if(w*h*8>weightBudget)break;const weights=maskWeightCache.get(layer,data,w,h,options.spatial,()=>{const values=new Float64Array(w*h);for(let i=0;i<values.length;i++)values[i]=calculateAlpha(layer,i);return values;});if(weights){cachedWeights.set(layer,weights);weightBudget-=weights.byteLength;}}
 const layerAlpha=(layer,index)=>cachedWeights.get(layer)?.[index]??calculateAlpha(layer,index);
 for(let i=0,j=0;i<data.length;i+=4,j+=3){let r=highPrecision?srgb(linear(data[i]/inputScale)*exp):exposureLUT[data[i]],g=highPrecision?srgb(linear(data[i+1]/inputScale)*exp):exposureLUT[data[i+1]],b=highPrecision?srgb(linear(data[i+2]/inputScale)*exp):exposureLUT[data[i+2]];
 if(p.matchMap){matchColor(p.matchMap,data[i]/inputScale,data[i+1]/inputScale,data[i+2]/inputScale,((i/4%w)+ox+.5)/fw,(Math.floor(i/4/w)+oy+.5)/fh,matchPixel);r=srgb(linear(matchPixel[0])*exp);g=srgb(linear(matchPixel[1])*exp);b=srgb(linear(matchPixel[2])*exp);}
 if(calibration){const lr=linear(r),lg=linear(g),lb=linear(b);r=srgb(calibration[0]*lr+calibration[1]*lg+calibration[2]*lb);g=srgb(calibration[3]*lr+calibration[4]*lg+calibration[5]*lb);b=srgb(calibration[6]*lr+calibration[7]*lg+calibration[8]*lb);}
 if(p.calibrationShadowTint){const shadow=shadowCalibration(linear(r),linear(g),linear(b),p.calibrationShadowTint);r=srgb(shadow[0]);g=srgb(shadow[1]);b=srgb(shadow[2]);}
 r*=1+p.temperature*.003+p.tint*.001;g*=1-p.tint*.002;b*=1-p.temperature*.003+p.tint*.001;
 let y=lum(r,g,b),delta=p.shadows/180*(1-smooth(.1,.65,y))*Math.min(1,y*8)+p.highlights/180*smooth(.4,1,y)+endpointDelta(y,p.whites,p.blacks);
 const toneCurve=highPrecision?extendedCurve:curveValue;const contrast=1+p.contrast/100;r=toneCurve((r-.5)*contrast+.5+delta,p.curve);g=toneCurve((g-.5)*contrast+.5+delta,p.curve);b=toneCurve((b-.5)*contrast+.5+delta,p.curve);
 if(usePointCurves){r=pointCurves.red(pointCurves.rgb(r));g=pointCurves.green(pointCurves.rgb(g));b=pointCurves.blue(pointCurves.rgb(b));}
 if(useHsl){const floor=highPrecision?Math.min(0,r,g,b):0,span=highPrecision?Math.max(1,Math.max(r,g,b)-floor):1;let [hh,ss,ll]=rgbToHsl(clamp((r-floor)/span),clamp((g-floor)/span),clamp((b-floor)/span));let dh=0,ds=0,dl=0,weight=0;
 for(let n=0;n<8;n++){const distance=Math.min(Math.abs(hh-centers[n]),360-Math.abs(hh-centers[n])),ww=Math.max(0,1-distance/(n<3?40:65));dh+=p.hsl[n].hue*ww;ds+=p.hsl[n].saturation*ww;dl+=p.hsl[n].lightness*ww;weight+=ww;}
 // Absolute chroma vanishes near white/black; HSL saturation alone does not.
 // Prevent selective lightness from amplifying tiny highlight hue variations.
 weight=Math.max(weight,1);[r,g,b]=hslToRgb(hh+dh/weight,clamp(ss*(1+(p.saturation+ds/weight)/100)),clamp(ll+dl/weight/180*ss*(1-Math.abs(2*ll-1))));if(highPrecision){r=r*span+floor;g=g*span+floor;b=b*span+floor;}
 }if(p.vibrance)[r,g,b]=vibranceColor(r,g,b,p.vibrance);if(p.blackWhite)r=g=b=blackWhiteValue(r,g,b,p.blackWhiteMix);if(useGrade){y=lum(r,g,b);gradingWeights(y,p.gradingBalance,p.gradingBlending,gradeWeights);const [sw,mw,hw]=gradeWeights;
 [r,g,b]=[r,g,b].map((v,c)=>v+grade.shadows[c]*sw+grade.midtones[c]*mw+grade.highlights[c]*hw);
 const gradeEV=(p.gradingShadowLuminance*sw+p.gradingMidtoneLuminance*mw+p.gradingHighlightLuminance*hw)/100;if(gradeEV){const factor=2**gradeEV;r=srgb(linear(r)*factor);g=srgb(linear(g)*factor);b=srgb(linear(b)*factor);}
 }if(p.gradingGlobalSaturation){r+=globalGrade[0];g+=globalGrade[1];b+=globalGrade[2];}if(p.gradingGlobalLuminance){r=srgb(linear(r)*globalGradeFactor);g=srgb(linear(g)*globalGradeFactor);b=srgb(linear(b)*globalGradeFactor);}if(lut){lutSampler(lut,r,g,b,lutPixel);r=lutPixel[0];g=lutPixel[1];b=lutPixel[2];}for(const layer of layers){if(layer.enabled===false)continue;const a=layerAlpha(layer,i/4);if(a===0)continue;if(layer.protect){r+=(data[i]/inputScale-r)*a;g+=(data[i+1]/inputScale-g)*a;b+=(data[i+2]/inputScale-b)*a;}else{const factor=2**((layer.exposure||0)*a);if(layer.exposureSpace==='srgb-v1'){r*=factor;g*=factor;b*=factor;}else if(factor!==1){r=srgb(linear(r)*factor);g=srgb(linear(g)*factor);b=srgb(linear(b)*factor);}
 const contrast=1+(layer.contrast||0)/100*a;r=.5+(r-.5)*contrast;g=.5+(g-.5)*contrast;b=.5+(b-.5)*contrast;
 const temperature=(layer.temperature||0)*a,tint=(layer.tint||0)*a;r*=1+temperature*.003+tint*.001;g*=1-tint*.002;b*=1-temperature*.003+tint*.001;
 const yy=lum(r,g,b),delta=(layer.shadows||0)*a/180*(1-smooth(.1,.65,yy))*Math.min(1,Math.max(0,yy)*8)+(layer.highlights||0)*a/180*smooth(.4,1,yy)+endpointDelta(yy,(layer.whites||0)*a,(layer.blacks||0)*a);r+=delta;g+=delta;b+=delta;
 const sy=lum(r,g,b),saturation=1+(layer.saturation||0)/100*a;r=sy+(r-sy)*saturation;g=sy+(g-sy)*saturation;b=sy+(b-sy)*saturation;if(layer.vibrance)[r,g,b]=vibranceColor(r,g,b,layer.vibrance*a);const curves=localCurves.get(layer);if(curves){r+=(curves.red(curves.rgb(r))-r)*a;g+=(curves.green(curves.rgb(g))-g)*a;b+=(curves.blue(curves.rgb(b))-b)*a;}}}
 if(p.colorRefinement){refineColor(p.colorRefinement,r,g,b,((i/4%w)+ox+.5)/fw,(Math.floor(i/4/w)+oy+.5)/fh,matchPixel);[r,g,b]=matchPixel;}
 if(p.referenceRecovery){recoverReference(p.referenceRecovery,((i/4%w)+ox+.5)/fw,(Math.floor(i/4/w)+oy+.5)/fh,r,g,b,matchPixel,recoverySource,i/4);[r,g,b]=matchPixel;}
 stage[j]=r;stage[j+1]=g;stage[j+2]=b;
 }

 // Spatial local detail is evaluated after local color, before global detail.
 // Filter the full neighborhood first, then blend only through the mask; never
 // filter a masked/cutout image, which would create seams around mask boundaries.
 if(layers.some(l=>!l.protect&&(l.dehaze||l.texture||l.clarity||l.noiseLuminance||l.noiseColor||l.sharpenAmount))){
  const baseline=layers.some(l=>l.protect)?stage.slice():null;
  for(const layer of layers){
   if(!layer.protect&&!layer.dehaze&&!layer.texture&&!layer.clarity&&!layer.noiseLuminance&&!layer.noiseColor&&!layer.sharpenAmount)continue;
   const weights=new Float32Array(w*h);let covered=false;for(let index=0;index<weights.length;index++){weights[index]=layerAlpha(layer,index);covered ||= weights[index]>0;}if(!covered)continue;
   if(!layer.protect){applyMaskedDetail(stage,w,h,layer,weights,options.pixelScale??1);continue;}
   for(let index=0,j=0;index<w*h;index++,j+=3){const a=weights[index];if(!a)continue;for(let c=0;c<3;c++)stage[j+c]+=(baseline[j+c]-stage[j+c])*a;}
  }
 }
 dehazeRGB(stage,w,h,p,options.pixelScale??1);
 denoiseRGB(stage,w,h,p,options.pixelScale??1);
 textureRGB(stage,w,h,p,options.pixelScale??1);
 clarityRGB(stage,w,h,p,options.pixelScale??1);
 sharpenRGB(stage,w,h,p,options.pixelScale??1);
 if(!p.highlightSoftness&&!p.grain&&!p.vignette){for(let i=0,j=0;i<data.length;i+=4,j+=3){out[i]=encodeOutput(data[i]/inputScale+(stage[j]-data[i]/inputScale)*amount,outputScale);out[i+1]=encodeOutput(data[i+1]/inputScale+(stage[j+1]-data[i+1]/inputScale)*amount,outputScale);out[i+2]=encodeOutput(data[i+2]/inputScale+(stage[j+2]-data[i+2]/inputScale)*amount,outputScale);out[i+3]=data[i+3]/inputScale*outputScale;}return out;}
 const blurred=p.highlightSoftness?blurRGB(stage,w,h,Math.max(1,Math.round(Math.max(fw,fh)/500))):null;
 for(let i=0,j=0;i<data.length;i+=4,j+=3){const x=i/4%w,y=Math.floor(i/4/w),l=lum(stage[j],stage[j+1],stage[j+2]);const softness=p.highlightSoftness/100*smooth(.45,.9,l);const edge=clamp((((x+ox)/fw-.5)**2+((y+oy)/fh-.5)**2)*2);let hash=Math.imul(x+ox+1,374761393)^Math.imul(y+oy+1,668265263);hash=Math.imul(hash^(hash>>>13),1274126177);const grain=((hash>>>0)/4294967295-.5)*p.grain/255*(.4+.6*(1-l));
 for(let c=0;c<3;c++){let v=stage[j+c];if(blurred)v+=(blurred[j+c]-v)*softness;v=v*(1-edge*p.vignette/100)+grain;out[i+c]=encodeOutput(data[i+c]/inputScale+(v-data[i+c]/inputScale)*amount,outputScale);}out[i+3]=data[i+3]/inputScale*outputScale;}return out;
}
export function statistics(data){const hist=Array.from({length:4},()=>Array(64).fill(0));let sum=0,sat=0,low=0,high=0,n=0;for(let i=0;i<data.length;i+=4){if(!data[i+3])continue;const r=data[i]/255,g=data[i+1]/255,b=data[i+2]/255,y=lum(r,g,b);for(let c=0;c<3;c++)hist[c][Math.min(63,data[i+c]>>2)]++;hist[3][Math.min(63,Math.floor(y*64))]++;sum+=y;sat+=rgbToHsl(r,g,b)[1];low+=y<.01;high+=y>.99;n++;}n=Math.max(1,n);return {mean:sum/n,saturation:sat/n,clippedShadows:low/n,clippedHighlights:high/n,histogram:hist};}
