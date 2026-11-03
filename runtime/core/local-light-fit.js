import {render} from './engine.js';
import {applyLocalLight} from './local-light.js';
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x)),lum=(r,g,b)=>.2126*r+.7152*g+.0722*b;
// Splat robust log gains into an x/y/luminance grid. Smooth numerator and mass
// together, never divide unsupported cells and propagate arbitrary huge gains.
function fitGrid(base,target,w,h,valid,longSide){
 const width=Math.max(2,Math.round(longSide*w/Math.max(w,h))),height=Math.max(2,Math.round(longSide*h/Math.max(w,h))),depth=8,n=width*height*depth;
 let mass=new Float32Array(n),sum=new Float32Array(n*3);const guideMax=1;
 for(let k=0;k<w*h;k++){
  if(k%7===0||valid&&!valid[k])continue;const i=k*4,r=base[i],g=base[i+1],b=base[i+2],tr=target[i]/255,tg=target[i+1]/255,tb=target[i+2]/255,y=lum(r,g,b),ty=lum(tr,tg,tb);
  if(y<.015||ty<.015||Math.max(r,g,b)>1.05||Math.max(tr,tg,tb)>.995)continue;
  const ev=clamp(Math.log2((ty+.01)/(y+.01)),-2,2),delta=[tr,tg,tb].map((v,c)=>ev+clamp(Math.log2((v+.03)/([r,g,b][c]+.03))-ev,-1.2,1.2));
  const coords=[clamp(((k%w)+.5)/w*width-.5,0,width-1),clamp((Math.floor(k/w)+.5)/h*height-.5,0,height-1),clamp(y,0,1)*(depth-1)],pos=coords.map(Math.floor),f=coords.map((v,c)=>v-pos[c]);
  for(let dz=0;dz<2;dz++)for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++){
   const q=(dx?f[0]:1-f[0])*(dy?f[1]:1-f[1])*(dz?f[2]:1-f[2]),j=((Math.min(depth-1,pos[2]+dz)*height+Math.min(height-1,pos[1]+dy))*width+Math.min(width-1,pos[0]+dx));mass[j]+=q;for(let c=0;c<3;c++)sum[j*3+c]+=q*delta[c];
  }
 }
 // Spatial smoothing at each luminance slice, keeping tonal boundaries separate.
 for(let pass=0;pass<2;pass++){
  const nm=new Float32Array(n),ns=new Float32Array(n*3);
  for(let z=0;z<depth;z++)for(let y=0;y<height;y++)for(let x=0;x<width;x++){
   const j=(z*height+y)*width+x;
   for(const [dx,dy,weight] of [[0,0,4],[-1,0,1],[1,0,1],[0,-1,1],[0,1,1]]){const xx=clamp(x+dx,0,width-1),yy=clamp(y+dy,0,height-1),a=(z*height+yy)*width+xx;nm[j]+=mass[a]*weight/8;for(let c=0;c<3;c++)ns[j*3+c]+=sum[a*3+c]*weight/8;}
  }mass=nm;sum=ns;
 }
 const values=new Float32Array(n*12);
 for(let j=0;j<n;j++)for(let c=0;c<3;c++)values[j*12+c*5]=2**(sum[j*3+c]/(mass[j]+.3));
 return {kind:'bilateral',width,height,depth,guideMax,values,strength:1};
}
export function fitLocalLight(source,target,w,h,baseline,{valid,highlights=.8,shadows=.8,scale='balanced',color=0,progress=()=>{}}={}){
 const params={...baseline.params};delete params.localLight;
 const base=render(source,w,h,params,{layers:baseline.layers,outputFormat:'float32'});
 progress({phase:'局部光影迁移',progress:.2});
 const value={version:1,strength:1,highlights,shadows,color,coarse:fitGrid(base,target,w,h,valid,8),fine:fitGrid(base,target,w,h,valid,{broad:12,balanced:24,local:32}[scale]||24)};
 let before=0,after=0,count=0,clipped=0;const out=new Float64Array(3);
 // Held-out pixels were excluded from fitting. Reject harmful or empty fields.
 for(let k=0;k<w*h;k++){if(k%7!==0||valid&&!valid[k])continue;const i=k*4;applyLocalLight(value,base[i],base[i+1],base[i+2],(k%w+.5)/w,(Math.floor(k/w)+.5)/h,out);for(let c=0;c<3;c++){const t=target[i+c]/255;before+=(base[i+c]-t)**2;after+=(out[c]-t)**2;clipped+=Number(out[c]>1&&t<.98&&base[i+c]<=1||out[c]<0&&t>.02&&base[i+c]>=0);}count+=3;}
 const accepted=count>=48&&after<before*.995&&clipped/count<.01;
 progress({phase:'局部光影迁移',progress:1});
 return {...baseline,params:accepted?{...params,localLight:value}:params,localLightAccepted:accepted,localLightError:count?Math.sqrt(after/count):0};
}
