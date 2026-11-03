import {lutCatalog} from './luts/catalog.js';
const metadata=new Map(lutCatalog.map(x=>[x.id,x])),cache=new Map(),pending=new Map();
export function validLut(id){return metadata.has(id);}
export function registerLut(id,table){const m=metadata.get(id);if(!m||!(table instanceof Float32Array)||table.length!==m.size**3*3||table.some(v=>!Number.isFinite(v)))throw Error('LUT 数据不完整');const value={...m,table};cache.set(id,value);return value;}
export function getLut(id){if(!id)return null;const value=cache.get(id);if(!value)throw Error('风格尚未加载，请重新选择风格');return value;}
export async function ensureLut(id){if(!id)return null;if(cache.has(id))return cache.get(id);if(!metadata.has(id))throw Error('未知的 LUT 风格');if(!pending.has(id))pending.set(id,(async()=>{const res=await fetch(new URL('./luts/'+id+'.bin',import.meta.url));if(!res.ok)throw Error('风格加载失败，请重试');return registerLut(id,new Float32Array(await res.arrayBuffer()));})().finally(()=>pending.delete(id)));return pending.get(id);}
// .cube ordering is red fastest, then green, then blue. Reuse result to avoid per-pixel allocation.
export function sampleLut(lut,r,g,b,result){
 const n=lut.size,d=lut.table,min=lut.domainMin,max=lut.domainMax;
 const rx=Math.max(0,Math.min(1,(r-min[0])/(max[0]-min[0])))*(n-1),gy=Math.max(0,Math.min(1,(g-min[1])/(max[1]-min[1])))*(n-1),bz=Math.max(0,Math.min(1,(b-min[2])/(max[2]-min[2])))*(n-1);
 const ix=Math.min(n-2,Math.floor(rx)),iy=Math.min(n-2,Math.floor(gy)),iz=Math.min(n-2,Math.floor(bz)),fx=rx-ix,fy=gy-iy,fz=bz-iz;
 const i=(iz*n*n+iy*n+ix)*3,dy=n*3,dz=n*n*3;
 for(let c=0;c<3;c++){const a=d[i+c]*(1-fx)+d[i+3+c]*fx,b=d[i+dy+c]*(1-fx)+d[i+dy+3+c]*fx,e=d[i+dz+c]*(1-fx)+d[i+dz+3+c]*fx,f=d[i+dz+dy+c]*(1-fx)+d[i+dz+dy+3+c]*fx;result[c]=(a*(1-fy)+b*fy)*(1-fz)+(e*(1-fy)+f*fy)*fz;}return result;
}

// SDR LUTs define no transform outside their domain. Extend their boundary
// correction with unit slope instead of clipping HDR/wide-gamut residuals.
// This preserves headroom and continuity, not an authored HDR look.
export function sampleLutExtended(lut,r,g,b,result){
 sampleLut(lut,r,g,b,result);
 const min=lut.domainMin,max=lut.domainMax;
 result[0]+=r-Math.max(min[0],Math.min(max[0],r));
 result[1]+=g-Math.max(min[1],Math.min(max[1],g));
 result[2]+=b-Math.max(min[2],Math.min(max[2],b));
 return result;
}
