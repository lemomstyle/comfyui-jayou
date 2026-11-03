export const bedConcepts=['bed','duvet','pillows','bed runner','bed sheet','blanket','objects on bed'];
export const windowConcepts=['window','outdoor view through window','outdoor view through open doorway'];
export const bedExclusions=['bedside table','objects on bedside table'];
export function subtractSelection(mask,excluded){
 if(mask.length!==excluded.length)throw Error('排除选区尺寸错误');
 const out=new Uint8Array(mask.length);for(let i=0;i<out.length;i++)out[i]=Math.round(mask[i]*(1-excluded[i]/255));return out;
}
export async function collectSelection(scope,prompt,request,{allowEmpty=false}={}){
 const concepts=scope==='bed'?bedConcepts:scope==='window'?windowConcepts:[prompt],masks=[];
 for(const word of concepts){const r=await request(word);if(!Array.isArray(r.masks))throw Error('分割响应缺少蒙版');masks.push(...r.masks);}
 if(!masks.length&&!allowEmpty)throw Error('未识别到所选区域');return masks;
}
function at(mask,w,h,x,y){x=Math.max(0,Math.min(w-1,x));y=Math.max(0,Math.min(h-1,y));const ix=Math.floor(x),iy=Math.floor(y),fx=x-ix,fy=y-iy;return mask[iy*w+ix]*(1-fx)*(1-fy)+mask[iy*w+Math.min(w-1,ix+1)]*fx*(1-fy)+mask[Math.min(h-1,iy+1)*w+ix]*(1-fx)*fy+mask[Math.min(h-1,iy+1)*w+Math.min(w-1,ix+1)]*fx*fy;}
export function pairedSelection(source,reference,w,h,rw,rh,alignment){
 if(source.length!==w*h||reference.length!==rw*rh)throw Error('双图选区尺寸错误');
 const out=source.slice(),m=alignment?.kind==='homography'?alignment.matrix:[1,0,alignment?.x||0,0,1,alignment?.y||0,0,0,1];
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){const u=(x+.5)/w,v=(y+.5)/h,d=m[6]*u+m[7]*v+m[8];if(Math.abs(d)<1e-8)continue;const xx=(m[0]*u+m[1]*v+m[2])/d,yy=(m[3]*u+m[4]*v+m[5])/d;if(xx<0||xx>1||yy<0||yy>1)continue;const k=y*w+x;out[k]=Math.max(out[k],Math.round(at(reference,rw,rh,xx*rw-.5,yy*rh-.5)));}return out;
}
// Radius is specified in ORIGINAL pixels, not in the segmentation thumbnail.
export function expandAndFeather(mask,w,h,originalWidth,originalHeight,pixels=2){
 if(!Number.isFinite(pixels)||pixels<1||pixels>3)throw Error('扩展与羽化需为 1–3 像素');
 if(mask.length!==w*h||w<1||h<1||originalWidth<w||originalHeight<h)throw Error('选区像素尺寸错误');
 const rx=pixels*w/originalWidth,ry=pixels*h/originalHeight,dilated=new Uint8Array(mask.length),out=new Uint8Array(mask.length),horizontal=new Float32Array(mask.length);
 const offsets=r=>{const values=[-r,0,r];for(let i=1;i<r;i++)values.push(-i,i);return values;},xs=offsets(rx),ys=offsets(ry);
 // Sample the entire expansion footprint, including fractional thumbnail endpoints.
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=0;for(const dx of xs)v=Math.max(v,at(mask,w,h,x+dx,y));horizontal[y*w+x]=v;}
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=0;for(const dy of ys)v=Math.max(v,at(horizontal,w,h,x,y+dy));dilated[y*w+x]=Math.round(v);}
 // A compact triangular feather with the same support radius as expansion.
 for(let y=0;y<h;y++)for(let x=0;x<w;x++)horizontal[y*w+x]=(at(dilated,w,h,x-rx,y)+2*dilated[y*w+x]+at(dilated,w,h,x+rx,y))/4;
 for(let y=0;y<h;y++)for(let x=0;x<w;x++)out[y*w+x]=Math.round((at(horizontal,w,h,x,y-ry)+2*horizontal[y*w+x]+at(horizontal,w,h,x,y+ry))/4);
 return out;
}

export function hasSelection(selection){return !!selection?.mask?.some(v=>v>0);}

// Independent signed expansion and feather radii in original-image pixels.
export function adjustMaskEdges(mask,w,h,originalWidth,originalHeight,expand=0,feather=0){
 if(mask.length!==w*h||w<1||h<1||!Number.isFinite(originalWidth)||!Number.isFinite(originalHeight)||originalWidth<1||originalHeight<1||!Number.isFinite(expand)||Math.abs(expand)>12||!Number.isFinite(feather)||feather<0||feather>12)throw Error('独立扩展与羽化参数无效');
 if(expand===0&&feather===0)return mask.slice();
 const offsets=r=>{const values=[-r,0,r];for(let i=1;i<r;i++)values.push(-i,i);return values;};
 const work=new Float32Array(mask.length);let result=mask.slice();
 if(expand){const xs=offsets(Math.abs(expand)*w/originalWidth),ys=offsets(Math.abs(expand)*h/originalHeight),fn=expand>0?Math.max:Math.min,initial=expand>0?0:255;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=initial;for(const dx of xs)v=fn(v,at(result,w,h,x+dx,y));work[y*w+x]=v;}
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=initial;for(const dy of ys)v=fn(v,at(work,w,h,x,y+dy));result[y*w+x]=Math.round(v);}
 }
 if(feather){const rx=feather*w/originalWidth,ry=feather*h/originalHeight;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++)work[y*w+x]=(at(result,w,h,x-rx,y)+2*result[y*w+x]+at(result,w,h,x+rx,y))/4;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++)result[y*w+x]=Math.round((at(work,w,h,x,y-ry)+2*work[y*w+x]+at(work,w,h,x,y+ry))/4);
 }
 return result;
}
