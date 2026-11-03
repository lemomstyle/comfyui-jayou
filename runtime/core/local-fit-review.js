import {gradientWeight} from './gradients.js';
// Assess center and transition independently: a useful center cannot pay for a halo.
export function reviewLocalChange(before,after,target,w,h,weights,gradient){
 const zones=Array.from({length:3},()=>({mass:0,before:0,after:0,harm:0}));let rough=0,edges=0;
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){
  const k=y*w+x,q=weights[k];if(q<.5)continue;
  const g=gradientWeight(gradient,(x+.5)/w,(y+.5)/h),z=zones[g>.6?0:g>.05?1:2],i=k*4;let old=0,next=0;
  for(let c=0;c<3;c++){old+=(before[i+c]-target[i+c]/255)**2/3;next+=(after[i+c]-target[i+c]/255)**2/3;}
  z.mass+=q;z.before+=old*q;z.after+=next*q;if(next>old+.0004)z.harm+=q;
  for(const j of [x+1<w?k+1:-1,y+1<h?k+w:-1]){if(j<0||weights[j]<.5)continue;let edge=0,added=0,expected=0;for(let c=0;c<3;c++){const d=before[j*4+c]-before[i+c],t=target[j*4+c]/255-target[i+c]/255;edge+=d*d;added+=(after[j*4+c]-after[i+c]-d)**2;expected+=(t-d)**2;}
   if(edge<.0003){rough+=Math.max(0,added-expected);edges++;}
  }
 }
 const mass=zones.reduce((s,z)=>s+z.mass,0),old=zones.reduce((s,z)=>s+z.before,0),next=zones.reduce((s,z)=>s+z.after,0);
 const zoneSafe=zones.every(z=>z.mass<8||(z.after<=z.before*1.025+z.mass*.00001&&z.harm/z.mass<.12));
 return {accepted:mass>0&&old-next>Math.max(old*.01,mass*.000002)&&zoneSafe&&rough/Math.max(1,edges)<.00001,improvement:(old-next)/Math.max(1,mass)};
}
export function localPenalty(l){return .00008*l.opacity*l.opacity*((l.exposure/.7)**2+(l.temperature/15)**2+(l.tint/15)**2+(l.contrast/20)**2+(l.saturation/20)**2);}
