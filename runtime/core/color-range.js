// Hue/saturation intersection sampled from the unadjusted image, not its result.
// This is an editable color-family selector, not semantic skin recognition.
function validateSingle(value){
 if(!value||typeof value!=='object')throw Error('颜色范围格式不合法');
 for(const [key,max] of [['hue',360],['width',180],['feather',180],['satMin',1],['satMax',1]])if(!Number.isFinite(value[key])||value[key]<0||value[key]>max)throw Error('颜色范围参数不合法：'+key);
 return Object.fromEntries(['hue','width','feather','satMin','satMax'].map(k=>[k,value[k]]));
}
export function validateColorRange(value){
 if(value==null)return null;
 const range=validateSingle(value);
 if(value.samples!==undefined){
  if(!Array.isArray(value.samples)||value.samples.length>4)throw Error('颜色范围最多支持 5 个样本');
  range.samples=value.samples.map(sample=>{if(sample?.samples!==undefined)throw Error('颜色样本不能嵌套');return validateSingle(sample);});
 }
 return range;
}
export function addColorSample(range,sample){
 const next=validateSingle(sample);if(!range)return next;
 const base=validateColorRange(range);if((base.samples?.length||0)>=4)throw Error('颜色范围最多支持 5 个样本，请先移除一个样本');
 return {...base,samples:[...(base.samples||[]),next]};
}
export function removeColorSample(range,index){
 const base=validateColorRange(range),all=base?[validateSingle(base),...(base.samples||[])]:[];
 if(!Number.isInteger(index)||index<0||index>=all.length)throw Error('颜色样本序号无效');
 all.splice(index,1);if(!all.length)return null;
 const [first,...samples]=all;return samples.length?{...first,samples}:first;
}
export function colorRangeWeight(range,r,g,b){
 if(!range)return 1;
 let weight=singleWeight(range,r,g,b);
 for(const sample of range.samples||[])weight=Math.max(weight,singleWeight(sample,r,g,b));
 return weight;
}
const smooth=x=>{const t=Math.max(0,Math.min(1,x));return t*t*(3-2*t);};
function singleWeight(range,r,g,b){
 // Shift negative extended RGB uniformly; avoid clipping over-white channels.
 const floor=Math.min(0,r,g,b);r-=floor;g-=floor;b-=floor;
 const max=Math.max(r,g,b),min=Math.min(r,g,b),delta=max-min,s=max>0?delta/max:0;
 const low=Math.min(range.satMin,range.satMax),high=Math.max(range.satMin,range.satMax);
 const sat=(low===0?1:smooth((s-low+.05)/.05))*(high===1?1:smooth((high+.05-s)/.05));
 if(!sat)return 0;
 if(range.width===180)return sat;
 if(delta<1e-10)return 0;
 let h=(max===r?(g-b)/delta:max===g?(b-r)/delta+2:(r-g)/delta+4)*60;h=(h+360)%360;
 const diff=Math.abs(h-range.hue),distance=Math.min(diff,360-diff);
 const hue=range.feather?smooth((range.width+range.feather-distance)/range.feather):Number(distance<=range.width);
 return hue*sat;
}
