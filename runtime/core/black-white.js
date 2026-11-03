export const blackWhiteNames=['红','橙','黄','绿','青','蓝','紫','洋红'];
const centers=[0,30,60,120,180,240,275,315];
const linear=x=>x<=.04045?x/12.92:((x+.055)/1.055)**2.4;
const encoded=x=>x<=.0031308?x*12.92:1.055*x**(1/2.4)-.055;
export function validateBlackWhite(p){
 const enabled=p.blackWhite===undefined?false:p.blackWhite,mix=p.blackWhiteMix===undefined?Array(8).fill(0):p.blackWhiteMix;
 if(typeof enabled!=='boolean'||!Array.isArray(mix)||mix.length!==8||mix.some(x=>!Number.isFinite(x)||x< -100||x>100))throw Error('黑白混合器参数不合法');
 return {blackWhite:enabled,blackWhiteMix:[...mix]};
}
// Luminance-preserving neutral conversion, with overlapping hue-band exposure
// offsets. Saturation gates the mixer so gray patches never follow color sliders.
export function blackWhiteValue(r,g,b,mix){
 const max=Math.max(r,g,b),min=Math.min(r,g,b),chroma=max-min;
 if(chroma<1e-12)return r;
 let hue=max===r?((g-b)/chroma+6)%6:max===g?(b-r)/chroma+2:(r-g)/chroma+4;hue*=60;
 let offset=0,total=0;
 for(let i=0;i<8;i++){const d=Math.abs(hue-centers[i]),weight=Math.max(0,1-Math.min(d,360-d)/(i<3?40:65));offset+=weight*mix[i];total+=weight;}
 const saturation=Math.min(1,chroma/Math.max(1e-12,max-Math.min(0,min))),ev=offset/Math.max(1e-12,total)/100*saturation;
 return encoded((.2126*linear(r)+.7152*linear(g)+.0722*linear(b))*2**ev);
}
