// Spatial illumination, not reference pixel replacement. Assets are small gain grids.
import {matchColor,transformMatchMap} from './same-image-map.js';
const lum=(r,g,b)=>.2126*r+.7152*g+.0722*b;
export function validateLocalLight(v){
 if(!v||v.version!==1)throw Error('局部光影版本无效');
 for(const k of ['strength','highlights','shadows','color'])if(!Number.isFinite(v[k])||v[k]<0||v[k]>1)throw Error('局部光影强度无效');
 for(const m of [v.coarse,v.fine]){if(m?.kind!=='bilateral')throw Error('局部光影网格无效');if(![m.width,m.height,m.depth].every(x=>Number.isInteger(x)&&x>=2&&x<=32)||m.guideMax!==1||m.strength!==1||!(Array.isArray(m.values)||m.values instanceof Float32Array)||m.values.length!==m.width*m.height*m.depth*12||m.values.some(x=>!Number.isFinite(x)||Math.abs(x)>16)||m.transform&&(!Array.isArray(m.transform)||m.transform.length!==6||!m.transform.every(Number.isFinite)))throw Error('局部光影网格数据无效');}return v;
}
export function applyLocalLight(v,r,g,b,x,y,out){
 out[0]=r;out[1]=g;out[2]=b;if(!v?.strength)return out;
 // Gain grids are queried using the current luminance; edges of different tone
 // do not share the same correction. No clipping of the original HDR texture.
 matchColor(v.coarse,r,g,b,x,y,out);const cr=out[0],cg=out[1],cb=out[2];
 matchColor(v.fine,r,g,b,x,y,out);const rr=.2*cr+.8*out[0],gg=.2*cg+.8*out[1],bb=.2*cb+.8*out[2];
 const y0=lum(r,g,b),y1=lum(rr,gg,bb);if(y0<=1e-6||y1<=1e-6){out[0]=r;out[1]=g;out[2]=b;return out;}
 const ev=Math.max(-2,Math.min(2,Math.log2(y1/y0))),gain=2**(ev*(ev>=0?v.highlights:v.shadows)*v.strength);
 // Chroma residual is luminance-neutral; toggling color does not change light.
 const chroma=v.color*v.strength,normalize=y0/y1;
 out[0]=(r+(rr*normalize-r)*chroma)*gain;out[1]=(g+(gg*normalize-g)*chroma)*gain;out[2]=(b+(bb*normalize-b)*chroma)*gain;return out;
}
export function transformLocalLight(v,inverse){return {...v,coarse:transformMatchMap(v.coarse,inverse),fine:transformMatchMap(v.fine,inverse)};}
