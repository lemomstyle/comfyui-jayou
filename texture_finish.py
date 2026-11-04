"""Reference-guided finishing; transforms the result, never copies reference detail."""
import cv2
import numpy as np
from .color import LUMA, smooth, rgb_hsl
from .common import resize, resize_to, chunks
from .fitting import correspondence


def blur(a, sigma):
    radius = max(1, int(np.ceil(3*sigma)))
    return cv2.GaussianBlur(a, (2*radius+1,)*2, sigma, borderType=cv2.BORDER_REFLECT_101)


def weighted_curve(y, target, weights):
    centers = np.linspace(0, 1, 33)
    means = np.zeros(33); masses = np.zeros(33)
    # Conditional robust corrections, rather than unpaired histogram matching.
    for i, center in enumerate(centers):
        w = weights*np.maximum(0, 1-np.abs(y-center)*32)
        if w.sum() < 12: continue
        residual = np.clip(target-y, -.12, .12)
        initial = float((w*residual).sum()/w.sum())
        w *= np.minimum(1, .035/np.maximum(.001, np.abs(residual-initial)))
        masses[i] = w.sum(); means[i] = (w*residual).sum()/(w.sum()+30)
    delta = cv2.GaussianBlur(means.astype(np.float32)[None], (7, 1), 1.1)[0]
    values = centers+np.clip(delta, -.07, .07)
    values[0], values[-1] = 0, 1
    # Enforce positive bounded slopes in both directions, keep endpoints fixed.
    for _ in range(4):
        for i in range(1, 32): values[i] = np.clip(values[i], values[i-1]+.35/32, values[i-1]+2/32)
        for i in range(31, 0, -1): values[i] = np.clip(values[i], values[i+1]-2/32, values[i+1]-.35/32)
    return values.astype(np.float32)


def adjust_tone(a, points):
    y = a @ LUMA
    target = np.interp(y, np.linspace(0, 1, len(points)), points).astype(np.float32)
    # Luminance delta retains neutral colors; soft range gates protect endpoints.
    return a+(target-y)[..., None]


def grid_sample(grid, shape, top, bottom):
    h, w = shape
    yy, xx = np.mgrid[top:bottom, :w]
    mx = ((xx+.5)/w*grid.shape[1]-.5).astype(np.float32)
    my = ((yy+.5)/h*grid.shape[0]-.5).astype(np.float32)
    return cv2.remap(grid, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def apply(a, recipe, strength, top=0, full_shape=None):
    if strength == 0 or recipe is None: return a.copy()
    shape = full_shape or a.shape[:2]
    q = adjust_tone(a, recipe['curve'])
    y = q @ LUMA
    sigma = max(.8, max(shape)/180)
    detail = y-blur(y, sigma)
    gain = grid_sample(recipe['detail'], shape, top, top+a.shape[0])
    # Mid-scale source detail only; cap strong edges and suppress smooth areas.
    signal = smooth(.0015, .012, np.abs(detail))
    local = np.clip(detail*gain*signal, -.025, .025)
    q += local[..., None]
    hue, _, _ = rgb_hsl(np.clip(q, 0, 1))
    indices = np.arange(9)*45
    chroma_gain = np.interp(hue, indices, np.r_[recipe['chroma'], recipe['chroma'][0]])
    luminance = q @ LUMA; chroma = np.ptp(q, axis=-1)
    chroma_gate = smooth(.025, .09, chroma)*(1-smooth(.35, .7, chroma))
    q = luminance[..., None]+(q-luminance[..., None])*(1+chroma_gain*chroma_gate)[..., None]
    confidence = grid_sample(recipe['confidence'], shape, top, top+a.shape[0])
    delta = (q-a)*(confidence*strength)[..., None]
    # Scale the entire RGB delta together to prevent new channel clipping/hue shifts.
    allowed = np.ones_like(a)
    np.divide(1-a, delta, out=allowed, where=delta > 1e-8)
    negative = delta < -1e-8
    np.divide(-a, delta, out=allowed, where=negative)
    limiter = np.clip(allowed.min(-1), 0, 1)
    return (a+delta*limiter[..., None]).astype(np.float32)


def objective(a, b, weights):
    ya, yb = a @ LUMA, b @ LUMA
    sigma = max(.8, max(a.shape[:2])/180)
    da, db = ya-blur(ya, sigma), yb-blur(yb, sigma)
    chroma_a, chroma_b = np.ptp(a, axis=-1), np.ptp(b, axis=-1)
    error = (ya-yb)**2+.7*(da-db)**2+.3*(chroma_a-chroma_b)**2
    return float((error*weights).sum()/max(float(weights.sum()), 1))


def fit(result, reference, original, valid):
    weights, mismatch = correspondence(original, reference, valid)
    rw, _ = correspondence(result, reference, valid)
    weights *= rw
    # Exclude edges and regions whose structure cannot support the comparison.
    weights *= (mismatch < .25)
    if np.count_nonzero(weights > .5) < 128: return None, {'accepted': False, 'reason': 'insufficient_correspondence'}
    held = np.arange(weights.size).reshape(weights.shape) % 7 == 0
    train, check = weights*(~held), weights*held
    y, target = result @ LUMA, reference @ LUMA
    curve = weighted_curve(y, target, train)
    toned = adjust_tone(result, curve); ty = toned @ LUMA
    sigma = max(.8, max(result.shape[:2])/180)
    da, db = ty-blur(ty, sigma), target-blur(target, sigma)
    # Smooth coefficient fields, not image pixels; require positively correlated detail.
    spatial_sigma = max(2, max(result.shape[:2])/30)
    mass = blur(train, spatial_sigma)+1e-5
    va = blur(da*da*train, spatial_sigma)/mass
    vb = blur(db*db*train, spatial_sigma)/mass
    cov = blur(da*db*train, spatial_sigma)/mass
    corr = cov/np.sqrt(va*vb+1e-10)
    noise = np.abs(ty-blur(ty, .65))
    noise_level = blur(noise*noise*train, spatial_sigma)/mass
    reliable = smooth(.4, .85, corr)*smooth(.00002, .0002, va)
    reliable *= 1-smooth(.6, 1.5, noise_level/(va+1e-6))
    gain = np.clip(np.sqrt(vb/(va+1e-6))-1, 0, .35)*reliable
    # Learn only missing colorfulness, avoiding neutral highlights and black pixels.
    hue, _, _ = rgb_hsl(np.clip(toned, 0, 1))
    ca, cb = np.ptp(toned, axis=-1), np.ptp(reference, axis=-1)
    chroma = []
    for center in np.arange(8)*45:
        distance = np.abs(hue-center); distance = np.minimum(distance, 360-distance)
        w = train*np.maximum(0, 1-distance/55)*(ca > .03)*(cb > .03)*(target > .08)*(target < .92)
        ratio = float((cb*w).sum()/max(float((ca*w).sum()), 1e-6))
        chroma.append(np.clip(ratio-1, 0, .22) if w.sum() > 30 else 0.)
    # Confidence stays spatial: bad TV/replaced content does not receive fitted corrections.
    confidence = blur((weights > .5).astype(np.float32), 2)
    confidence *= (valid > .99)
    recipe = {'curve': curve, 'detail': gain.astype(np.float32),
              'chroma': np.array(chroma, np.float32), 'confidence': confidence.astype(np.float32)}
    before = objective(result, reference, check)
    candidate = apply(result, recipe, 1)
    after = objective(candidate, reference, check)
    # Require quality gain without trading away RGB accuracy.
    rgb_before = float((np.square(result-reference).mean(-1)*check).sum()/max(float(check.sum()), 1))
    rgb_after = float((np.square(candidate-reference).mean(-1)*check).sum()/max(float(check.sum()), 1))
    accepted = after < before*.995 and rgb_after <= rgb_before*1.02+1e-6
    return (recipe if accepted else None), {'accepted': bool(accepted), 'before': before, 'after': after}


def finish(output, reference, original, overlap, strength):
    if strength <= 0: return {'accepted': False, 'reason': 'zero_strength'}
    a = resize(output, 512); b = resize_to(reference, a.shape[:2]); source = resize_to(original, a.shape[:2])
    valid = resize_to(overlap, a.shape[:2])
    recipe, diagnostics = fit(a, b, source, valid)
    if recipe is None: return diagnostics
    h, w = output.shape[:2]; radius = max(1, int(np.ceil(3*max(.8, max(h, w)/180))))
    # Read from an unchanged input: in-place strips would otherwise feed previous
    # enhanced rows into the next strip's neighborhood and create seams.
    result = np.empty_like(output)
    for top, bottom in chunks(h, w):
        lo, hi = max(0, top-radius), min(h, bottom+radius)
        strip = apply(output[lo:hi], recipe, strength, lo, (h, w))
        base = output[top:bottom]
        gate = np.clip(overlap[top:bottom], 0, 1)[..., None]
        result[top:bottom] = base+(strip[top-lo:bottom-lo]-base)*gate
    output[:] = result
    return diagnostics
