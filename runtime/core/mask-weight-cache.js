// Exact-content cache: pixel masks and sampled color/luminance selections must
// invalidate even when a caller mutates one element in place.
const equal=(a,b)=>{if(!a||!b)return a===b;if(a.constructor!==b.constructor||a.length!==b.length)return false;for(let i=0;i<a.length;i++)if(a[i]!==b[i])return false;return true;};
export function createMaskWeightCache(maxBytes=64*1024*1024,maxEntries=8){
 const entries=[];let bytes=0,hits=0,misses=0;
 return {
  get(layer,data,w,h,spatial,build){
   const dependent=(!layer.gradient&&!layer.mask)||(layer.range&&layer.range!=='all')||layer.luminanceRange||layer.colorRange;
   const source=dependent?data:null,mask=layer.mask||null;
   const key=JSON.stringify([w,h,spatial?.x||0,spatial?.y||0,spatial?.width||w,spatial?.height||h,data.constructor.name,layer.type,layer.gradient,layer.range,layer.luminanceRange,layer.colorRange,layer.inverted,layer.opacity??1,layer.brushOps,layer.postBrushOps]);
   const index=entries.findIndex(e=>e.key===key&&equal(e.source,source)&&equal(e.mask,mask));
   if(index>=0){hits++;const [e]=entries.splice(index,1);entries.push(e);return e.weights;}
   misses++;const size=w*h*8+(source?.byteLength||0)+(mask?(mask.byteLength??mask.length*8):0);if(size>maxBytes)return null;
   const weights=build();while(entries.length&&(bytes+size>maxBytes||entries.length>=maxEntries))bytes-=entries.shift().size;
   entries.push({key,source:source?.slice()||null,mask:mask?.slice()||null,weights,size});bytes+=size;return weights;
  },
  clear(){entries.length=0;bytes=hits=misses=0;},
  info(){return {bytes,hits,misses,entries:entries.length,maxBytes};}
 };
}
export const maskWeightCache=createMaskWeightCache();
