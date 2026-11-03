// Eight-connected background avoids sealing diagonal gaps between fine objects.
export function exteriorBackground(mask,w,h){
 const seen=new Uint8Array(w*h),queue=new Int32Array(w*h);let read=0,end=0;
 const add=k=>{if(!seen[k]&&mask[k]<128){seen[k]=1;queue[end++]=k;}};
 for(let x=0;x<w;x++){add(x);add((h-1)*w+x);}for(let y=0;y<h;y++){add(y*w);add(y*w+w-1);}
 while(read<end){const k=queue[read++],x=k%w,y=Math.floor(k/w);for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const xx=x+dx,yy=y+dy;if(xx>=0&&xx<w&&yy>=0&&yy<h)add(yy*w+xx);}}
 return seen;
}
export function fillSmallHoles(mask,w,h,maxArea){
 const out=mask.slice();if(maxArea<1)return out;
 const seen=exteriorBackground(mask,w,h),queue=new Int32Array(w*h);
 for(let k=0;k<mask.length;k++)if(mask[k]<128&&!seen[k]){let read=0,end=1;queue[0]=k;seen[k]=1;let rim=255;
  while(read<end){const j=queue[read++],x=j%w,y=Math.floor(j/w);for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){const xx=x+dx,yy=y+dy;if(xx<0||xx>=w||yy<0||yy>=h)continue;const q=yy*w+xx;if(mask[q]>=128){rim=Math.min(rim,mask[q]);continue;}if(!seen[q]){seen[q]=1;queue[end++]=q;}}}
  if(end<=maxArea)for(let i=0;i<end;i++)out[queue[i]]=rim;
 }
 return out;
}
