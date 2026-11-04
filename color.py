"""Float RGB transforms; no interpreter bridge or image-sized LUT cache."""
import numpy as np

LUMA = np.array([.2126, .7152, .0722], np.float32)
CENTERS = np.array([0, 30, 60, 120, 180, 240, 275, 315])
KEYS = ['exposure', 'temperature', 'tint', 'contrast', 'shadows', 'highlights',
        'whites', 'blacks', 'saturation', 'vibrance']
LIMITS = [(-3, 3), (-50, 50), (-50, 50), (-60, 60), (-60, 60), (-60, 60),
          (-100, 100), (-100, 100), (-70, 70), (-100, 100)]


def smooth(a, b, x):
    t = np.clip((x-a)/(b-a), 0, 1)
    return t*t*(3-2*t)


def linear(x):
    return np.where(x <= .04045, x/12.92, ((np.maximum(x, 0)+.055)/1.055)**2.4)


def encoded(x):
    return np.where(x <= .0031308, x*12.92, 1.055*np.maximum(x, 0)**(1/2.4)-.055)


def defaults():
    return dict(basic=np.zeros(10), hsl=np.zeros((8, 3)),
                curves=np.tile(np.linspace(0, 1, 5), (4, 1)))


def curve(v, points):
    if np.array_equal(points, np.linspace(0, 1, 5)):
        return v
    d = np.diff(points)*4
    m = np.zeros(5); m[0], m[-1] = d[0], d[-1]
    for i in range(1, 4):
        if d[i-1]*d[i] > 0:
            m[i] = 2/(1/d[i-1]+1/d[i])
    ix = np.clip(np.floor(v*4).astype(np.int32), 0, 3)
    t = np.clip(v*4-ix, 0, 1); t2 = t*t; t3 = t2*t
    out = ((2*t3-3*t2+1)*points[ix]+(t3-2*t2+t)*.25*m[ix]
           +(-2*t3+3*t2)*points[ix+1]+(t3-t2)*.25*m[ix+1])
    return np.where(v < 0, points[0]+v*m[0],
                    np.where(v > 1, points[-1]+(v-1)*m[-1], out))


def rgb_hsl(a):
    hi, lo = a.max(-1), a.min(-1); d = hi-lo
    l = (hi+lo)*.5
    s = np.divide(d, 1-np.abs(2*l-1), out=np.zeros_like(d), where=d > 1e-10)
    safe = np.maximum(d, 1e-10)
    r, g, b = np.moveaxis(a, -1, 0)
    h = np.where(hi == r, ((g-b)/safe+6) % 6,
                 np.where(hi == g, (b-r)/safe+2, (r-g)/safe+4))*60
    return np.where(d > 1e-10, h, 0), s, l


def hsl_rgb(h, s, l):
    h = h % 360; c = (1-np.abs(2*l-1))*s
    x = c*(1-np.abs((h/60) % 2-1)); z = np.zeros_like(c)
    sector = (h/60).astype(np.int32)
    r = np.choose(sector, [c, x, z, z, x, c])
    g = np.choose(sector, [x, c, c, x, z, z])
    b = np.choose(sector, [z, z, x, c, c, x])
    return np.stack([r, g, b], -1)+(l-c*.5)[..., None]


def render(a, p):
    exposure, temp, tint, contrast, shadows, highlights, whites, blacks, sat, vib = p['basic']
    q = encoded(linear(a)*2**exposure) if exposure else a.copy()
    q *= np.array([1+temp*.003+tint*.001, 1-tint*.002, 1-temp*.003+tint*.001])
    y = q @ LUMA
    delta = (shadows/180*(1-smooth(.1, .65, y))*np.minimum(1, y*8)
             +highlights/180*smooth(.4, 1, y)+whites/400*smooth(.5, 1, y)
             +blacks/400*(1-smooth(0, .5, y)))
    q = (q-.5)*(1+contrast/100)+.5+delta[..., None]
    q = curve(q, p['curves'][0])
    for c in range(3):
        q[..., c] = curve(q[..., c], p['curves'][c+1])
    if sat or np.any(p['hsl']):
        floor = np.minimum(0, q.min(-1)); span = np.maximum(1, q.max(-1)-floor)
        h, s, l = rgb_hsl(np.clip((q-floor[..., None])/span[..., None], 0, 1))
        dh = np.zeros_like(h); ds = dh.copy(); dl = dh.copy(); mass = dh.copy()
        for i, center in enumerate(CENTERS):
            d = np.abs(h-center); d = np.minimum(d, 360-d)
            w = np.maximum(0, 1-d/(40 if i < 3 else 65))
            dh += p['hsl'][i, 0]*w; ds += p['hsl'][i, 1]*w
            dl += p['hsl'][i, 2]*w; mass += w
        mass = np.maximum(1, mass)
        q = hsl_rgb(h+dh/mass, np.clip(s*(1+(sat+ds/mass)/100), 0, 1),
                    np.clip(l+dl/mass/180*s*(1-np.abs(2*l-1)), 0, 1))
        q = q*span[..., None]+floor[..., None]
    if vib:
        hi, lo = q.max(-1), q.min(-1)
        saturation = np.clip((hi-lo)/np.maximum(hi, 1e-10), 0, 1)
        h, _, _ = rgb_hsl(np.clip(q/np.maximum(1, hi)[..., None], 0, 1))
        warm = 1-smooth(20, 55, np.abs(h-30))
        factor = 1+vib/100*(1-saturation)**2*((1-.5*warm) if vib > 0 else 1)
        y = q @ LUMA; q = y[..., None]+(q-y[..., None])*factor[..., None]
    return q.astype(np.float32)
