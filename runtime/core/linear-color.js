// Extended sRGB transfer. Negative values and values above 1 remain representable.
export const linearToEncoded=v=>v<=.0031308?12.92*v:1.055*v**(1/2.4)-.055;
export const encodedToLinear=v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4;
