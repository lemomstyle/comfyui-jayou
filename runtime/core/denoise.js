// Separable joint-bilateral denoising in luminance and two chroma differences.
// Original float RGB guides both passes; no range clipping or synthesized detail.
export const denoiseDefaults={noiseLuminance:0,noiseLuminanceDetail:50,noiseColor:0,noiseColorDetail:50};
export function denoiseSupport(p,scale=1){return p.noiseLuminance>0||p.noiseColor>0?Math.ceil(2*Math.max(.15,1.5*scale)):0;}
const rangeWeights=Float64Array.from({length:2049},(_,i)=>Math.exp(-i/128));
function weightAt(x){if(x>=16)return 0;const position=x*128,index=Math.floor(position),f=position-index;return rangeWeights[index]*(1-f)+rangeWeights[index+1]*f;}
export function denoiseRGB(rgb,w,h,p,scale=1){
 if(!p.noiseLuminance&&!p.noiseColor)return rgb;
 const sigma=Math.max(.15,1.5*scale),radius=denoiseSupport(p,scale),kernel=new Float32Array(radius*2+1);
 for(let k=-radius;k<=radius;k++)kernel[k+radius]=Math.exp(-k*k/(2*sigma*sigma));
 const original=new Float32Array(rgb.length),tmp=new Float32Array(rgb.length),filtered=new Float32Array(rgb.length);
 for(let i=0;i<rgb.length;i+=3){const y=.2126*rgb[i]+.7152*rgb[i+1]+.0722*rgb[i+2];original[i]=y;original[i+1]=rgb[i]-y;original[i+2]=rgb[i+2]-y;}
 const lAmount=p.noiseLuminance/100,cAmount=p.noiseColor/100;
 const lSigma=.004+.14*lAmount*(1-.9*p.noiseLuminanceDetail/100),cSigma=.008+.22*cAmount*(1-.85*p.noiseColorDetail/100);
 const li=1/(2*lSigma*lSigma),ci=1/(2*cSigma*cSigma),yi=1/(2*.12*.12);
 for(let axis=0;axis<2;axis++){
  const input=axis?tmp:original,output=axis?filtered:tmp;
  for(let y=0;y<h;y++)for(let x=0;x<w;x++){
   const i=(y*w+x)*3;let ls=0,cs=0,lv=0,cr=0,cb=0;
   for(let k=-radius;k<=radius;k++){
    const sx=axis?x:Math.max(0,Math.min(w-1,x+k)),sy=axis?Math.max(0,Math.min(h-1,y+k)):y,j=(sy*w+sx)*3;
    const dy=original[j]-original[i],dr=original[j+1]-original[i+1],db=original[j+2]-original[i+2],spatial=kernel[k+radius];
    if(lAmount){const weight=spatial*weightAt(dy*dy*li);ls+=weight;lv+=input[j]*weight;}
    if(cAmount){const weight=spatial*weightAt(dy*dy*yi+(dr*dr+db*db)*ci);cs+=weight;cr+=input[j+1]*weight;cb+=input[j+2]*weight;}
   }
   output[i]=lAmount?lv/ls:original[i];output[i+1]=cAmount?cr/cs:original[i+1];output[i+2]=cAmount?cb/cs:original[i+2];
  }
 }
 for(let i=0;i<rgb.length;i+=3){const y=original[i]+(filtered[i]-original[i])*lAmount,cr=original[i+1]+(filtered[i+1]-original[i+1])*cAmount,cb=original[i+2]+(filtered[i+2]-original[i+2])*cAmount;
  rgb[i]=y+cr;rgb[i+2]=y+cb;rgb[i+1]=(y-.2126*rgb[i]-.0722*rgb[i+2])/.7152;
 }
 return rgb;
}
