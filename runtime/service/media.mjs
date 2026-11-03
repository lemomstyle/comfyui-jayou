import https from 'node:https';
import http from 'node:http';
import {lookup} from 'node:dns/promises';
import {isIP} from 'node:net';
export function publicIPv4(ip){
 if(isIP(ip)!==4)return false;
 const [a,b,c]=ip.split('.').map(Number);
 return !(a===0||a===10||a===127||a>=224||(a===169&&b===254)||(a===172&&b>=16&&b<=31)||(a===192&&(b===168||b===0||b===2))||(a===100&&b>=64&&b<=127)||(a===198&&(b===18||b===19||b===51&&c===100))||(a===203&&b===0&&c===113));
}
export function publicIP(ip){
 if(isIP(ip)===4)return publicIPv4(ip);
 if(isIP(ip)!==6)return false;
 const [first,second]=ip.toLowerCase().split(':').map(x=>parseInt(x||'0',16));
 // Global unicast only; exclude mapped/local, transition and documentation ranges.
 return first>=0x2000&&first<0x3fff&&first!==0x2002&&!(first===0x2001&&(second<=0x1ff||second===0xdb8));
}
export function webURL(raw){const u=new URL(raw);if(!['https:','http:'].includes(u.protocol)||u.username||u.password||u.port&&u.port!=='80'&&u.port!=='443')throw Error('不支持的图片地址');return u;}
export function imageMime(bytes){if(bytes[0]===255&&bytes[1]===216&&bytes[2]===255)return 'image/jpeg';if(bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])))return 'image/png';if(bytes.toString('ascii',0,4)==='RIFF'&&bytes.toString('ascii',8,12)==='WEBP')return 'image/webp';throw Error('图片格式不支持，请选择 JPEG、PNG 或 WebP');}
// Pin the resolved public address on every hop: no redirects or DNS rebinding into localhost.
export async function fetchImage(raw,hops=0){
 if(hops>3)throw Error('图片跳转过多');const u=webURL(raw);
 const addresses=await lookup(u.hostname,{all:true,verbatim:true});
 const addr=addresses.find(x=>publicIP(x.address));if(!addr||addresses.some(x=>!publicIP(x.address)))throw Error('图片地址不可访问');
 return new Promise((resolve,reject)=>{
 const transport=u.protocol==='https:'?https:http;
 const req=transport.get(u,{family:addr.family,lookup:(_host,options,cb)=>options.all?cb(null,[addr]):cb(null,addr.address,addr.family),headers:{'User-Agent':'SizhuiColorLab/0.2 (reference image preview)','Accept':'image/jpeg,image/png,image/webp'}},res=>{
 if([301,302,303,307,308].includes(res.statusCode)){res.resume();if(!res.headers.location)return reject(Error('图片跳转缺少地址'));fetchImage(new URL(res.headers.location,u).href,hops+1).then(resolve,reject);return;}
 if(res.statusCode!==200){res.resume();return reject(Error(`图片源返回 ${res.statusCode}，可打开来源页查看`));}
 if(Number(res.headers['content-length'])>10_000_000){res.destroy();return reject(Error('图片大于 10 MB'));}
 let size=0;const chunks=[];res.on('data',b=>{size+=b.length;if(size>10_000_000){req.destroy(Error('图片大于 10 MB'));return;}chunks.push(b);});res.on('end',()=>{try{const bytes=Buffer.concat(chunks);resolve({bytes,mime:imageMime(bytes)});}catch(e){reject(e);}});res.on('error',reject);
 });const timer=setTimeout(()=>req.destroy(Error('图片加载超时，请尝试其他作品')),20000);req.on('close',()=>clearTimeout(timer));req.on('error',reject);
 });
}
