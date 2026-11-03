// Local structure comparison, deliberately independent of exposure/color errors.
// This is an experimental correspondence mask, not semantic sky/window detection.
const scale=p=>p instanceof Float32Array?1:p instanceof Uint16Array?65535:255;
const clamp=v=>Math.max(0,Math.min(1,v));
export function boxMean(data,w,h,r=5){
 const stride=w+1,integral=new Float64Array(stride*(h+1)),out=new Float32Array(w*h);
 for(let y=0;y<h;y++){let row=0;for(let x=0;x<w;x++){row+=data[y*w+x];integral[(y+1)*stride+x+1]=integral[y*stride+x+1]+row;}}
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){const l=Math.max(0,x-r),t=Math.max(0,y-r),rr=Math.min(w,x+r+1),b=Math.min(h,y+r+1);out[y*w+x]=(integral[b*stride+rr]-integral[t*stride+rr]-integral[b*stride+l]+integral[t*stride+l])/((rr-l)*(b-t));}return out;
}
export function analyzeMatchRegions(source,target,w,h,valid){
 const n=w*h,s=scale(source),a=new Float32Array(n),b=new Float32Array(n),aa=new Float32Array(n),bb=new Float32Array(n),ab=new Float32Array(n);
 for(let k=0;k<n;k++){const i=k*4;a[k]=clamp((.2126*source[i]+.7152*source[i+1]+.0722*source[i+2])/s);b[k]=(.2126*target[i]+.7152*target[i+1]+.0722*target[i+2])/255;aa[k]=a[k]*a[k];bb[k]=b[k]*b[k];ab[k]=a[k]*b[k];}
 const ma=boxMean(a,w,h),mb=boxMean(b,w,h),va=boxMean(aa,w,h),vb=boxMean(bb,w,h),cov=boxMean(ab,w,h),different=new Uint8Array(n),clipped=new Uint8Array(n);
 for(let k=0;k<n;k++){if(!valid[k])continue;const sa=Math.sqrt(Math.max(0,va[k]-ma[k]**2)),sb=Math.sqrt(Math.max(0,vb[k]-mb[k]**2)),corr=(cov[k]-ma[k]*mb[k])/(sa*sb+1e-7);clipped[k]=a[k]>.965&&b[k]<.94?1:0;
  // Both flat surfaces with different brightness are not changed content.
  if((sb>.035&&sa<.008)||(sa>.025&&sb>.025&&corr<.15)||clipped[k])different[k]=1;
 }
 const density=boxMean(different,w,h,3),excluded=new Uint8Array(n),fitValid=new Uint8Array(valid),contentMask=new Uint8Array(n),seen=new Uint8Array(n),queue=new Int32Array(n);let count=0,regions=0;
 for(let k=0;k<n;k++)if(valid[k]&&density[k]>.42){excluded[k]=1;fitValid[k]=0;count++;}
 for(let seed=0;seed<n;seed++){if(!excluded[seed]||seen[seed])continue;let head=0,tail=1,clipCount=0,boundary=0,goodBoundary=0;queue[0]=seed;seen[seed]=1;
  while(head<tail){const k=queue[head++],x=k%w,y=Math.floor(k/w);clipCount+=clipped[k];for(const j of [x?k-1:-1,x<w-1?k+1:-1,y?k-w:-1,y<h-1?k+w:-1]){if(j<0||!valid[j]){boundary++;continue;}if(!excluded[j]){boundary++;goodBoundary++;continue;}if(!seen[j]){seen[j]=1;queue[tail++]=j;}}}
  // Keep only substantial bounded components. Geometry still needs user review.
  if(tail>=Math.max(32,n*.004)&&tail<n*.4&&goodBoundary/Math.max(1,boundary)>.8){regions++;for(let j=0;j<tail;j++)contentMask[queue[j]]=255;}
 }
 // Feather inside the component; never bridge invalid alignment borders.
 const soft=boxMean(contentMask,w,h,2);for(let k=0;k<n;k++)contentMask[k]=valid[k]&&contentMask[k]?Math.round(soft[k]):0;
 return {fitValid,excluded,contentMask,regions,excludedFraction:count/n};
}
export function recoveryToneField(rendered,target,w,h){
 const out=new Float32Array(w*h*3);for(let c=0;c<3;c++){const d=new Float32Array(w*h);for(let k=0;k<d.length;k++)d[k]=target[k*4+c]/255-rendered[k*4+c];const smooth=boxMean(d,w,h,8);for(let k=0;k<d.length;k++)out[k*3+c]=smooth[k];}return out;
}
