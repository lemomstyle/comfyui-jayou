// Exact-content bounded LRU. No sampled hashes: even a single changed pixel
// invalidates neighborhood calculations. Each retained source is immutable.
export function createPixelCache(maxBytes,maxEntries=2){
 const entries=[];let bytes=0,hits=0,misses=0;
 return {
  get(pixels,key){const index=entries.findIndex(e=>{if(e.key!==key||e.source.length!==pixels.length)return false;for(let i=0;i<pixels.length;i++)if(e.source[i]!==pixels[i])return false;return true;});
   if(index<0){misses++;return null;}hits++;const [entry]=entries.splice(index,1);entries.push(entry);return entry.value;
  },
  put(pixels,key,value,valueBytes){const size=pixels.byteLength+valueBytes;if(size>maxBytes)return false;
   while(entries.length&&(bytes+size>maxBytes||entries.length>=maxEntries)){bytes-=entries.shift().size;}
   entries.push({source:pixels.slice(),key,value,size});bytes+=size;return true;
  },
  clear(){entries.length=0;bytes=0;hits=0;misses=0;},
  info(){return {bytes,entries:entries.length,hits,misses,maxBytes};}
 };
}
