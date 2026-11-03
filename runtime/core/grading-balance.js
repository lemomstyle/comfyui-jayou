const smooth=(a,b,x)=>{const t=Math.max(0,Math.min(1,(x-a)/(b-a)));return t*t*(3-2*t);};
// Default endpoints exactly match the legacy grading pipeline. Balance shifts
// tone membership; blending widens transitions without multiplying total tint.
export function gradingWeights(luminance,balance=0,blending=50,out=new Float64Array(3)){
 const y=luminance+balance/200,shadowSpread=.52*(blending-50)/200,highlightSpread=.55*(blending-50)/200;
 const shadow=1-smooth(.08-shadowSpread,.6+shadowSpread,y),highlight=smooth(.4-highlightSpread,.95+highlightSpread,y),mid=Math.max(0,1-shadow-highlight),total=shadow+mid+highlight;
 const divisor=balance===0&&blending===50?1:total;out[0]=shadow/divisor;out[1]=mid/divisor;out[2]=highlight/divisor;return out;
}
