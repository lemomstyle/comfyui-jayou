// Reference-to-source correspondence; source pixels never get warped.
export function validateAlignment(value){
 if(!value||value.kind!=='homography'||!Array.isArray(value.matrix)||value.matrix.length!==9||!value.matrix.every(Number.isFinite))throw Error('图像对齐数据无效');
 const m=value.matrix,d=m[0]*(m[4]*m[8]-m[5]*m[7])-m[1]*(m[3]*m[8]-m[5]*m[6])+m[2]*(m[3]*m[7]-m[4]*m[6]);
 if(Math.abs(d)<1e-8||value.matrix.some(v=>Math.abs(v)>100))throw Error('图像对齐变换不稳定');
 return value;
}
export function warpReference(target,tw,th,w,h,alignment){
 const m=validateAlignment(alignment).matrix,pixels=new Uint8ClampedArray(w*h*4),valid=new Uint8Array(w*h);
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
  const u=(x+.5)/w,v=(y+.5)/h,d=m[6]*u+m[7]*v+m[8];if(Math.abs(d)<1e-8)continue;
  const xx=(m[0]*u+m[1]*v+m[2])/d*tw-.5,yy=(m[3]*u+m[4]*v+m[5])/d*th-.5;
  if(xx<-.001||yy<-.001||xx>tw-1+.001||yy>th-1+.001)continue;
  const sx=Math.max(0,Math.min(tw-1,xx)),sy=Math.max(0,Math.min(th-1,yy)),ix=Math.floor(sx),iy=Math.floor(sy),fx=sx-ix,fy=sy-iy,k=y*w+x;
  for(let c=0;c<3;c++)pixels[k*4+c]=target[(iy*tw+ix)*4+c]*(1-fx)*(1-fy)+target[(iy*tw+Math.min(tw-1,ix+1))*4+c]*fx*(1-fy)+target[(Math.min(th-1,iy+1)*tw+ix)*4+c]*(1-fx)*fy+target[(Math.min(th-1,iy+1)*tw+Math.min(tw-1,ix+1))*4+c]*fx*fy;
  pixels[k*4+3]=255;valid[k]=1;
 }
 return {pixels,valid};
}
export function alignmentPreview(original,reference,w,h,alignment){
 const warped=warpReference(reference.pixels,reference.width,reference.height,w,h,alignment),out=new Uint8ClampedArray(original);
 for(let i=0;i<w*h;i++)if(warped.valid[i])for(let c=0;c<3;c++)out[i*4+c]=(original[i*4+c]+warped.pixels[i*4+c])/2;
 return out;
}
