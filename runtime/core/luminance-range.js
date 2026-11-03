export function validateLuminanceRange(value){
 if(value==null)return null;
 if(typeof value!=='object'||!['low','high','feather'].every(k=>Number.isFinite(value[k])&&value[k]>=0&&value[k]<=1))throw Error('亮度范围须为 0–1 的端点和过渡宽度');
 return {low:value.low,high:value.high,feather:value.feather};
}
const smooth=x=>{const t=Math.max(0,Math.min(1,x));return t*t*(3-2*t);};
export function luminanceRangeWeight(range,r,g,b){
 if(!range)return 1;
 const low=Math.min(range.low,range.high),high=Math.max(range.low,range.high),f=range.feather,y=.2126*r+.7152*g+.0722*b;
 // Endpoints 0 and 1 leave extended negative / over-white pixels selected.
 const lower=low===0?1:f?smooth((y-low+f)/f):Number(y>=low);
 const upper=high===1?1:f?smooth((high+f-y)/f):Number(y<=high);
 return lower*upper;
}
