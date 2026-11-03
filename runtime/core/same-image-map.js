// Serializable, spatially aware match assets. Kept separate from editable sliders.
const checked=new WeakSet(),checkedValues=new WeakSet();
export function validateMatchMap(map){
 if(!map)return null;if(checked.has(map))return map;
 if(!['lut','residual','bilateral'].includes(map.kind)||!Number.isFinite(map.strength)||map.strength<0||map.strength>1)throw Error('匹配映射不合法');
 if(map.transform&&(!Array.isArray(map.transform)||map.transform.length!==6||map.transform.some(v=>!Number.isFinite(v)||Math.abs(v)>1e6)))throw Error('匹配坐标不合法');
 const n=map.kind==='lut'?map.size**3*3:map.kind==='bilateral'?map.width*map.height*map.depth*12:map.width*map.height*3;
 if(map.kind==='bilateral'&&(![map.width,map.height,map.depth].every(v=>Number.isInteger(v)&&v>=2&&v<=16)||!Number.isFinite(map.guideMax)||map.guideMax<1||map.guideMax>16))throw Error('颜色变换网格不合法');
 if(map.kind==='lut'&&(!Number.isInteger(map.size)||map.size<2||map.size>33))throw Error('颜色映射尺寸不合法');
 if(map.kind==='residual'&&(!Number.isInteger(map.width)||!Number.isInteger(map.height)||map.width<1||map.height<1||map.width>2048||map.height>2048))throw Error('像素映射尺寸不合法');
 if(!(Array.isArray(map.values)||map.values instanceof Float32Array)||map.values.length!==n||(!checkedValues.has(map.values)&&map.values.some(v=>!Number.isFinite(v)||Math.abs(v)>1e6)))throw Error('匹配映射数据不合法');
 checked.add(map);checkedValues.add(map.values);return map;
}
export function matchColor(map,r,g,b,x,y,out){
 out[0]=r;out[1]=g;out[2]=b;if(!map||!map.strength)return out;
 const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
 if(map.kind==='residual'||map.kind==='bilateral'){
  if(map.transform){const m=map.transform,nx=m[0]*x+m[1]*y+m[2];y=m[3]*x+m[4]*y+m[5];x=nx;}if(x<0||x>1||y<0||y>1)return out;
  if(map.kind==='bilateral'){
   const px=clamp(x*map.width-.5,0,map.width-1),py=clamp(y*map.height-.5,0,map.height-1),pz=clamp((.2126*r+.7152*g+.0722*b)/map.guideMax,0,1)*(map.depth-1),ix=Math.floor(px),iy=Math.floor(py),iz=Math.floor(pz),fx=px-ix,fy=py-iy,fz=pz-iz;
   let rr=0,gg=0,bb=0;
   for(let dz=0;dz<2;dz++)for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++){
    const weight=(dx?fx:1-fx)*(dy?fy:1-fy)*(dz?fz:1-fz),j=((Math.min(iz+dz,map.depth-1)*map.height+Math.min(iy+dy,map.height-1))*map.width+Math.min(ix+dx,map.width-1))*12,v=map.values;
    rr+=weight*(v[j]*r+v[j+1]*g+v[j+2]*b+v[j+3]);gg+=weight*(v[j+4]*r+v[j+5]*g+v[j+6]*b+v[j+7]);bb+=weight*(v[j+8]*r+v[j+9]*g+v[j+10]*b+v[j+11]);
   }
   out[0]+=(rr-r)*map.strength;out[1]+=(gg-g)*map.strength;out[2]+=(bb-b)*map.strength;return out;
  }
  const px=clamp(x*map.width-.5,0,map.width-1),py=clamp(y*map.height-.5,0,map.height-1),ix=Math.floor(px),iy=Math.floor(py),fx=px-ix,fy=py-iy;
  for(let dy=0;dy<2;dy++)for(let dx=0;dx<2;dx++){const weight=(dx?fx:1-fx)*(dy?fy:1-fy)*map.strength,offset=(Math.min(iy+dy,map.height-1)*map.width+Math.min(ix+dx,map.width-1))*3;for(let c=0;c<3;c++)out[c]+=map.values[offset+c]*weight;}
 }else{
  const n=map.size,coords=[r,g,b].map(v=>clamp(v,0,1)*(n-1)),base=coords.map(Math.floor),f=coords.map((v,i)=>v-base[i]);
  for(let z=0;z<2;z++)for(let y=0;y<2;y++)for(let x=0;x<2;x++){const weight=(x?f[0]:1-f[0])*(y?f[1]:1-f[1])*(z?f[2]:1-f[2])*map.strength,offset=((Math.min(base[2]+z,n-1)*n+Math.min(base[1]+y,n-1))*n+Math.min(base[0]+x,n-1))*3;for(let c=0;c<3;c++)out[c]+=map.values[offset+c]*weight;}
 }
 return out;
}
// Keep spatial differences attached when the user crops, rotates or straightens.
export function transformMatchMap(map,inverse){
 if(!['residual','bilateral'].includes(map?.kind))return map;
 const m=map.transform||[1,0,0,0,1,0],convert=p=>({x:m[0]*p.x+m[1]*p.y+m[2],y:m[3]*p.x+m[4]*p.y+m[5]}),o=convert(inverse({x:0,y:0})),x=convert(inverse({x:1,y:0})),y=convert(inverse({x:0,y:1}));
 return {...map,transform:[x.x-o.x,y.x-o.x,o.x,x.y-o.y,y.y-o.y,o.y]};
}
