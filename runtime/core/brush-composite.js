// Ordered additions/subtractions remain independent of the procedural base.
// max/min brush operations reduce to clamp(base, lower, upper) per pixel.
export function validateBrushOps(ops){
 if(ops==null)return;
 if(!Array.isArray(ops)||ops.length>100000)throw Error('画笔操作数据非法');
 for(const o of ops)if(!o||!['from','to'].every(k=>o[k]&&['x','y'].every(c=>Number.isFinite(o[k][c])&&Math.abs(o[k][c])<1e6))||!Number.isFinite(o.radius)||o.radius<=0||o.radius>1e6||!Number.isFinite(o.hardness)||o.hardness<0||o.hardness>1||typeof o.erase!=='boolean'||['flow','density'].some(k=>o[k]!==undefined&&(!Number.isFinite(o[k])||o[k]<0||o[k]>1))||(o.densityMode!==undefined&&o.densityMode!=='target')||(o.densityMode==='target'&&o.flow===undefined)||(o.flow!==undefined&&(typeof o.stroke!=='string'||!o.stroke||o.stroke.length>128)))throw Error('画笔操作参数非法');
}
export function rasterizeBrushBounds(ops,w,h,spatial={},previous=null){
 if(!ops?.length)return null;
 if(ops.some(o=>o.flow!==undefined)||previous?.scale)return rasterizeFlowBounds(ops,w,h,spatial,previous);
 const lower=previous?previous.lower.slice():new Float32Array(w*h),upper=previous?previous.upper.slice():new Float32Array(w*h).fill(1),fw=spatial.width||w,fh=spatial.height||h,ox=spatial.x||0,oy=spatial.y||0;
 for(const o of ops){const x0=o.from.x*fw-ox,y0=o.from.y*fh-oy,x1=o.to.x*fw-ox,y1=o.to.y*fh-oy,r=o.radius*fw,dx=x1-x0,dy=y1-y0,length=dx*dx+dy*dy;
  const left=Math.max(0,Math.floor(Math.min(x0,x1)-r)),right=Math.min(w,Math.ceil(Math.max(x0,x1)+r)),top=Math.max(0,Math.floor(Math.min(y0,y1)-r)),bottom=Math.min(h,Math.ceil(Math.max(y0,y1)+r));
  for(let y=top;y<bottom;y++)for(let x=left;x<right;x++){const t=length?Math.max(0,Math.min(1,((x+.5-x0)*dx+(y+.5-y0)*dy)/length)):0,d=Math.hypot(x+.5-x0-t*dx,y+.5-y0-t*dy)/r;if(d>=1)continue;const a=d<=o.hardness?1:(1-d)/Math.max(1e-8,1-o.hardness),i=y*w+x;if(o.erase){lower[i]=Math.min(lower[i],1-a);upper[i]=Math.min(upper[i],1-a);}else{lower[i]=Math.max(lower[i],a);upper[i]=Math.max(upper[i],a);}}
 }
 return {lower,upper};
}
export function composeBrushWeight(base,bounds,index){return bounds?Math.max(bounds.lower[index],Math.min(bounds.upper[index],bounds.scale?bounds.scale[index]*base+bounds.bias[index]:base)):base;}
export function appendBrushOp(layer,from,to,settings){return [...(layer.brushOps||[]),{from:{...from},to:{...to},...settings}];}

// Exact serialized content keys survive worker structured cloning. Cached arrays
// are read-only to callers; appending strokes always copies before extending.
export function createBrushBoundsCache({maxBytes=32*1024*1024,maxEntries=8}={}){
 const entries=new Map();let bytes=0,hits=0,misses=0,incremental=0;
 function cached(ops,w,h,spatial={}){
  if(!ops?.length)return null;
  const geometry=JSON.stringify([w,h,spatial.width||w,spatial.height||h,spatial.x||0,spatial.y||0]),serialized=JSON.stringify(ops),key=geometry+serialized;
  const found=entries.get(key);if(found){hits++;entries.delete(key);entries.set(key,found);return found.value;}
  misses++;let prefix=null;
  for(const item of entries.values())if(item.geometry===geometry&&item.count<ops.length&&(!(ops[item.count]?.flow!==undefined&&ops[item.count].stroke===ops[item.count-1]?.stroke)||canContinueFlow(item.value.active,ops[item.count]))&&(!prefix||item.count>prefix.count)&&serialized.startsWith(item.serialized.slice(0,-1)+','))prefix=item;
  const value=prefix?rasterizeBrushBounds(ops.slice(prefix.count),w,h,spatial,prefix.value):rasterizeBrushBounds(ops,w,h,spatial);if(prefix)incremental++;
  const size=value.lower.byteLength+value.upper.byteLength+(value.scale?.byteLength||0)+(value.bias?.byteLength||0)+activeBytes(value.active)+(key.length+serialized.length+geometry.length)*2;
  if(size<=maxBytes&&maxEntries>0){while(entries.size&&(bytes+size>maxBytes||entries.size>=maxEntries)){const first=entries.keys().next().value;bytes-=entries.get(first).size;entries.delete(first);}entries.set(key,{value,size,geometry,serialized,count:ops.length});bytes+=size;}
  return value;
 }
 cached.clear=()=>{entries.clear();bytes=0;};
 cached.stats=()=>({bytes,entries:entries.size,hits,misses,incremental,maxBytes});
 return cached;
}
export const brushBounds=createBrushBoundsCache();

// New flow strokes compose as clamp(scale * base + bias, lower, upper).
// This representation also preserves old max/min brush recipes exactly.
function rasterizeFlowBounds(ops,w,h,spatial,previous){
 const n=w*h,lower=previous?previous.lower.slice():new Float32Array(n),upper=previous?previous.upper.slice():new Float32Array(n).fill(1),scale=previous?.scale?previous.scale.slice():new Float32Array(n).fill(1),bias=previous?.bias?previous.bias.slice():new Float32Array(n),fw=spatial.width||w,fh=spatial.height||h,ox=spatial.x||0,oy=spatial.y||0;
 const coverage=new Float32Array(n),seed=canContinueFlow(previous?.active,ops[0])?previous.active:null;let active=null;
 if(seed){restorePatch(seed,lower,upper,scale,bias,coverage,w);}

 for(let start=0;start<ops.length;){const first=ops[start],flow=first.flow!==undefined;let end=start+1;
 if(flow)while(end<ops.length&&ops[end].flow!==undefined&&ops[end].stroke===first.stroke&&ops[end].erase===first.erase&&ops[end].densityMode===first.densityMode&&(first.densityMode!=='target'||ops[end].density===first.density))end++;
 if(start||!seed)coverage.fill(0);let left=start===0&&seed?seed.left:w,right=start===0&&seed?seed.right:0,top=start===0&&seed?seed.top:h,bottom=start===0&&seed?seed.bottom:0;
 for(let k=start;k<end;k++){const o=ops[k],x0=o.from.x*fw-ox,y0=o.from.y*fh-oy,x1=o.to.x*fw-ox,y1=o.to.y*fh-oy,r=o.radius*fw,dx=x1-x0,dy=y1-y0,length=dx*dx+dy*dy;
 const l=Math.max(0,Math.floor(Math.min(x0,x1)-r)),rr=Math.min(w,Math.ceil(Math.max(x0,x1)+r)),t=Math.max(0,Math.floor(Math.min(y0,y1)-r)),bb=Math.min(h,Math.ceil(Math.max(y0,y1)+r));left=Math.min(left,l);right=Math.max(right,rr);top=Math.min(top,t);bottom=Math.max(bottom,bb);
 for(let y=t;y<bb;y++)for(let x=l;x<rr;x++){const q=length?Math.max(0,Math.min(1,((x+.5-x0)*dx+(y+.5-y0)*dy)/length)):0,d=Math.hypot(x+.5-x0-q*dx,y+.5-y0-q*dy)/r;if(d>=1)continue;const a=d<=o.hardness?1:(1-d)/Math.max(1e-8,1-o.hardness),i=y*w+x;coverage[i]=flow?Math.min(o.densityMode==='target'?1:(o.density??1),coverage[i]+(1-coverage[i])*a*o.flow):a;}
 }
 if(end===ops.length&&flow)active=capturePatch(first,{left:Math.max(0,Math.min(w,left)),right:Math.max(0,Math.min(w,right)),top:Math.max(0,Math.min(h,top)),bottom:Math.max(0,Math.min(h,bottom))},lower,upper,scale,bias,coverage,w);
 for(let y=top;y<bottom;y++)for(let x=left;x<right;x++){const i=y*w+x,a=coverage[i];if(!a)continue;if(flow){const m=1-a,target=first.densityMode==='target'?(first.erase?1-(first.density??1):(first.density??1)):(first.erase?0:1),b=target*a;scale[i]*=m;bias[i]=bias[i]*m+b;lower[i]=lower[i]*m+b;upper[i]=upper[i]*m+b;}else if(first.erase){lower[i]=Math.min(lower[i],1-a);upper[i]=Math.min(upper[i],1-a);}else{lower[i]=Math.max(lower[i],a);upper[i]=Math.max(upper[i],a);}}
 start=end;
 }
 return {lower,upper,scale,bias,active};
}

// Fixed spacing in image coordinates keeps flow independent of pointer event rate.
export function createBrushSampler(start,radius,aspect=1){
 if(!Number.isFinite(radius)||radius<=0||!Number.isFinite(aspect)||aspect<=0)throw Error('画笔采样尺寸非法');
 let last={...start},remaining=radius*.25;const spacing=remaining;
 return {move(to){const dx=to.x-last.x,dy=(to.y-last.y)/aspect,length=Math.hypot(dx,dy),out=[];if(length){let distance=remaining;while(distance<=length+1e-12){const t=Math.min(1,distance/length);out.push({x:last.x+(to.x-last.x)*t,y:last.y+(to.y-last.y)*t,...(start.pressure===undefined?{}:{pressure:last.pressure+(to.pressure-last.pressure)*t})});distance+=spacing;}remaining=distance-length;}last={...to};return out;}};
}

// Retain only the active stroke's bounding rectangle, not another full master.
function canContinueFlow(active,op){return !!active&&op?.flow!==undefined&&active.stroke===op.stroke&&active.erase===op.erase&&active.densityMode===op.densityMode&&(op.densityMode!=='target'||active.density===op.density);}
function activeBytes(active){return active?['lower','upper','scale','bias','coverage'].reduce((sum,key)=>sum+active[key].byteLength,0):0;}
function capturePatch(op,rect,lower,upper,scale,bias,coverage,w){
 const width=Math.max(0,rect.right-rect.left),height=Math.max(0,rect.bottom-rect.top),out={...rect,stroke:op.stroke,erase:op.erase,densityMode:op.densityMode,density:op.densityMode==='target'?op.density:undefined};
 for(const [key,array] of Object.entries({lower,upper,scale,bias,coverage})){const data=new Float32Array(width*height);for(let y=0;y<height;y++)data.set(array.subarray((y+rect.top)*w+rect.left,(y+rect.top)*w+rect.left+width),y*width);out[key]=data;}return out;
}
function restorePatch(active,lower,upper,scale,bias,coverage,w){
 const width=Math.max(0,active.right-active.left),height=Math.max(0,active.bottom-active.top);
 for(const [key,array] of Object.entries({lower,upper,scale,bias,coverage}))for(let y=0;y<height;y++)array.set(active[key].subarray(y*width,(y+1)*width),(y+active.top)*w+active.left);
}
