const clamp=v=>Math.max(0,Math.min(1,v));
export function validateGradient(g){if(!g||!['linear','radial'].includes(g.kind))throw Error('渐变类型错误');for(const k of ['x1','y1','x2','y2'])if(!Number.isFinite(g[k])||g[k]<-8||g[k]>9)throw Error('渐变坐标超出范围');if(!Number.isFinite(g.feather)||g.feather<0||g.feather>1)throw Error('羽化范围错误');if(g.angle!==undefined&&!Number.isFinite(g.angle))throw Error('旋转角度错误');if(g.aspect!==undefined&&(!Number.isFinite(g.aspect)||g.aspect<=0||g.aspect>100))throw Error('图像比例错误');if(typeof g.invert!=='boolean')throw Error('渐变反向值错误');if(g.kind==='linear'?Math.hypot(g.x2-g.x1,g.y2-g.y1)<.001:Math.min(Math.abs(g.x2-g.x1),Math.abs(g.y2-g.y1))<.001)throw Error('渐变范围过小');return g;}
// Linear: start fully selected, end unselected. Radial: bounding ellipse, feather inward.
export function gradientWeight(g,x,y){let v;if(g.kind==='linear'){const aspect=g.aspect||1,dx=(g.x2-g.x1)*aspect,dy=g.y2-g.y1;v=1-clamp(((x-g.x1)*aspect*dx+(y-g.y1)*dy)/(dx*dx+dy*dy));}else{const aspect=g.aspect||1,a=(g.angle||0)*Math.PI/180,dx=(x-(g.x1+g.x2)/2)*aspect,dy=y-(g.y1+g.y2)/2,rx=Math.abs(g.x2-g.x1)/2*aspect,ry=Math.abs(g.y2-g.y1)/2,d=Math.hypot((dx*Math.cos(a)+dy*Math.sin(a))/rx,(-dx*Math.sin(a)+dy*Math.cos(a))/ry);v=clamp((1-d)/Math.max(.001,g.feather));}v=v*v*(3-2*v);return g.invert?1-v:v;}

// Geometry is independent of segmentation; a bitmap, when present, restricts it.
export function geometryCoverage(layer,index,x,y,fallback){
 const bitmap=layer.mask?layer.mask[index]/255:1;
 return layer.gradient?gradientWeight(layer.gradient,x,y)*bitmap:layer.mask?bitmap:fallback();
}
export function prepareRegionGeometry(region,width,height){
 if(!region.gradient)return {...region};
 const gradient={...region.gradient,angle:region.gradient.angle??0,aspect:width/height};
 validateGradient(gradient);
 return {...region,gradient};
}
