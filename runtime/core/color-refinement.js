// A small, serializable post-grade transform; no reference pixels are retained.
import {validateMatchMap,matchColor,transformMatchMap} from './same-image-map.js';
export function validateColorRefinement(value){
 if(!value||value.version!==1||!Number.isFinite(value.strength)||value.strength<0||value.strength>1)throw Error('精细颜色变换参数无效');
 if(value.global?.kind!=='lut'||value.local&&value.local.kind!=='bilateral')throw Error('精细颜色变换类型无效');
 validateMatchMap(value.global);if(value.local)validateMatchMap(value.local);return value;
}
export function refineColor(value,r,g,b,x,y,out){
 matchColor(value.global,r,g,b,x,y,out);
 if(value.local)matchColor(value.local,out[0],out[1],out[2],x,y,out);
 const s=value.strength;out[0]=r+(out[0]-r)*s;out[1]=g+(out[1]-g)*s;out[2]=b+(out[2]-b)*s;return out;
}
export function transformColorRefinement(value,inverse){return {...value,...(value.local?{local:transformMatchMap(value.local,inverse)}:{})};}
