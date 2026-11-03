export const curveChannels=['rgb','red','green','blue'];
export const identityPoints=()=>[{x:0,y:0},{x:1,y:1}];
export const defaultPointCurves=()=>Object.fromEntries(curveChannels.map(c=>[c,identityPoints()]));
export function validatePointCurve(points){
 if(!Array.isArray(points)||points.length<2||points.length>32||points[0]?.x!==0||points.at(-1)?.x!==1)throw Error('点曲线须包含 2–32 个节点，首尾输入为 0 和 1');
 return points.map((p,i)=>{if(!p||!Number.isFinite(p.x)||!Number.isFinite(p.y)||p.x<0||p.x>1||p.y<0||p.y>1||(i&&p.x-points[i-1].x<.001))throw Error('曲线节点坐标无效或间距过小');return {x:p.x,y:p.y};});
}
// Shape-preserving cubic Hermite interpolation. No overshoot between anchor values.
export function compilePointCurve(points){
 const p=validatePointCurve(points),n=p.length,h=[],d=[],m=[];
 for(let i=0;i<n-1;i++){h[i]=p[i+1].x-p[i].x;d[i]=(p[i+1].y-p[i].y)/h[i];}
 m[0]=d[0];m[n-1]=d[n-2];
 for(let i=1;i<n-1;i++){if(d[i-1]*d[i]<=0)m[i]=0;else{const w1=2*h[i]+h[i-1],w2=h[i]+2*h[i-1];m[i]=(w1+w2)/(w1/d[i-1]+w2/d[i]);}}
 if(p.every(v=>v.x===v.y))return v=>v;
 return v=>{
  if(v<0)return p[0].y+v*m[0];if(v>1)return p[n-1].y+(v-1)*m[n-1];
  let i=0;while(i<n-2&&v>p[i+1].x)i++;
  const t=(v-p[i].x)/h[i],t2=t*t,t3=t2*t;
  return (2*t3-3*t2+1)*p[i].y+(t3-2*t2+t)*h[i]*m[i]+(-2*t3+3*t2)*p[i+1].y+(t3-t2)*h[i]*m[i+1];
 };
}
