// Single-channel sampling: no RGBA canvases, color conversion or GPU readback.
// Area integration on reduction prevents thin selections disappearing; expansion
// interpolates pixel centres. Tile coordinates share the full-image sampling grid.
function axis(source,full,start,length){
 const scale=source/full,all=[];
 for(let j=0;j<length;j++){
  const taps=[];
  if(scale>1){const a=(start+j)*scale,b=(start+j+1)*scale;for(let i=Math.floor(a);i<Math.ceil(b);i++){const weight=(Math.min(b,i+1)-Math.max(a,i))/scale;if(weight>0)taps.push([Math.min(source-1,Math.max(0,i)),weight]);}}
  else{const c=(start+j+.5)*scale-.5,lo=Math.floor(c),f=c-lo;taps.push([Math.max(0,Math.min(source-1,lo)),1-f]);if(f)taps.push([Math.max(0,Math.min(source-1,lo+1)),f]);}
  all.push(taps);
 }return all;
}
export function sampleMask(mask,sw,sh,{width,height,x=0,y=0,fullWidth=width,fullHeight=height}){
 if(!(mask instanceof Uint8Array||mask instanceof Uint8ClampedArray)||![sw,sh,width,height,fullWidth,fullHeight].every(v=>Number.isSafeInteger(v)&&v>0)||![x,y].every(v=>Number.isSafeInteger(v)&&v>=0)||mask.length!==sw*sh||x+width>fullWidth||y+height>fullHeight)throw Error('蒙版采样尺寸错误');
 if(sw===width&&sh===height&&x===0&&y===0&&fullWidth===width&&fullHeight===height)return mask;
 const xs=axis(sw,fullWidth,x,width),ys=axis(sh,fullHeight,y,height),first=ys[0][0][0],last=ys.at(-1).at(-1)[0];
 const horizontal=new Float32Array(width*(last-first+1));
 for(let row=first;row<=last;row++){const base=row*sw,out=(row-first)*width;for(let col=0;col<width;col++){const taps=xs[col];let value=mask[base+taps[0][0]]*taps[0][1];for(let k=1;k<taps.length;k++)value+=mask[base+taps[k][0]]*taps[k][1];horizontal[out+col]=value;}}
 const result=new Uint8Array(width*height);
 for(let row=0;row<height;row++){
  const taps=ys[row],base=row*width,a=(taps[0][0]-first)*width,wa=taps[0][1];
  if(taps.length<=2){const b=taps.length===2?(taps[1][0]-first)*width:a,wb=taps.length===2?taps[1][1]:0;for(let col=0;col<width;col++)result[base+col]=Math.round(horizontal[a+col]*wa+horizontal[b+col]*wb);}
  else for(let col=0;col<width;col++){let value=0;for(let k=0;k<taps.length;k++)value+=horizontal[(taps[k][0]-first)*width+col]*taps[k][1];result[base+col]=Math.round(value);}
 }
 return result;
}
