"""Small bilateral coefficient grids; original image texture stays full size."""
import numpy as np
from .color import LUMA
from .common import interrupted


def coords(rgb, shape, top=0, full_shape=None):
    depth, gh, gw = shape
    h, w = rgb.shape[:2]; fh, fw = full_shape or (h, w)
    yy, xx = np.mgrid[top:top+h, :w]
    return [np.clip((xx+.5)/fw*gw-.5, 0, gw-1),
            np.clip((yy+.5)/fh*gh-.5, 0, gh-1),
            np.clip(rgb @ LUMA, 0, 1)*(depth-1)]


def corners(rgb, shape, top=0, full_shape=None):
    depth, gh, gw = shape
    cs = coords(rgb, shape, top, full_shape); base = [np.floor(v).astype(np.int32) for v in cs]
    frac = [v-i for v, i in zip(cs, base)]
    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                idx = ((np.minimum(depth-1, base[2]+dz)*gh+np.minimum(gh-1, base[1]+dy))*gw
                       +np.minimum(gw-1, base[0]+dx)).ravel()
                q = ((frac[0] if dx else 1-frac[0])*(frac[1] if dy else 1-frac[1])
                     *(frac[2] if dz else 1-frac[2])).ravel()
                yield idx, q


def sample(grid, rgb, top=0, full_shape=None):
    flat = grid.reshape(-1, *grid.shape[3:])
    result = np.zeros((rgb.shape[0]*rgb.shape[1],)+grid.shape[3:], np.float32)
    for idx, q in corners(rgb, grid.shape[:3], top, full_shape):
        result += flat[idx]*q.reshape((-1,)+(1,)*(grid.ndim-3))
    return result.reshape(rgb.shape[:2]+grid.shape[3:])


def smooth_grid(values, mass, iterations=6):
    for _ in range(iterations):
        total = np.zeros_like(values); count = np.zeros(mass.shape)
        for axis in range(3):
            for direction in [-1, 1]:
                target = [slice(None)]*3; other = [slice(None)]*3
                target[axis] = slice(1, None) if direction == 1 else slice(None, -1)
                other[axis] = slice(None, -1) if direction == 1 else slice(1, None)
                total[tuple(target)] += values[tuple(other)]; count[tuple(target)] += 1
        anchor = np.clip(mass, 1, 30)
        extra = (None,)*(values.ndim-3)
        values = (values*anchor[(...,)+extra]+total)/(anchor+count)[(...,)+extra]
    return values.astype(np.float32)


def fit_affine(a, b, weights, size=5):
    shape = (8, size, size); n = np.prod(shape)
    v = np.concatenate([a.reshape(-1, 3), np.ones((a.shape[0]*a.shape[1], 1))], axis=1)
    target = b.reshape(-1, 3)
    valid = weights.ravel().copy(); valid[np.arange(valid.size) % 7 == 0] = 0
    global_a = v.T @ (valid[:, None]*v); global_b = v.T @ (valid[:, None]*target)
    ridge = max(.01, float(valid.sum())*.0001)
    identity = np.eye(4, 3)
    prior = np.linalg.solve(global_a+np.eye(4)*ridge, global_b+ridge*identity)
    grid = np.broadcast_to(prior, (*shape, 4, 3)).copy()
    for iteration in range(3):
        interrupted()
        weight = valid.copy()
        if iteration:
            out = apply_affine(grid, a)
            error = np.sqrt(np.square(out-b).mean(-1)).ravel()
            weight *= np.minimum(1, .04/np.maximum(.0001, error))
        matrix = np.zeros((n, 4, 4)); rhs = np.zeros((n, 4, 3)); mass = np.zeros(n)
        for idx, factor in corners(a, shape):
            q = factor*weight
            mass += np.bincount(idx, weights=q, minlength=n)
            for row in range(4):
                for col in range(4):
                    matrix[:, row, col] += np.bincount(idx, weights=q*v[:, row]*v[:, col], minlength=n)
                for c in range(3):
                    rhs[:, row, c] += np.bincount(idx, weights=q*v[:, row]*target[:, c], minlength=n)
        ridge = np.maximum(2, mass*.04)
        grid = np.linalg.solve(matrix+ridge[:, None, None]*np.eye(4), rhs+ridge[:, None, None]*prior)
        grid = smooth_grid(grid.reshape(*shape, 4, 3), mass.reshape(shape))
    return grid


def apply_affine(grid, a, top=0, full_shape=None):
    # Only strip-sized coefficient arrays are expanded, never the full image.
    coeff = sample(grid, a, top, full_shape)
    return (np.einsum('...i,...ic->...c', a, coeff[..., :3, :])+coeff[..., 3, :]).astype(np.float32)


def fit_gain(a, b, valid, side):
    h, w = a.shape[:2]; shape = (8, max(2, round(side*h/max(h, w))), max(2, round(side*w/max(h, w))))
    n = np.prod(shape); ya, yb = a @ LUMA, b @ LUMA
    ok = (ya > .015)&(yb > .015)&(a.max(-1) < 1.05)&(b.max(-1) < .995)
    weight = valid*ok; weight.ravel()[np.arange(weight.size) % 7 == 0] = 0
    ev = np.clip(np.log2((yb+.01)/(np.maximum(ya, 0)+.01)), -2, 2)
    delta = ev[..., None]+np.clip(np.log2((b+.03)/(np.maximum(a, 0)+.03))-ev[..., None], -1.2, 1.2)
    mass = np.zeros(n); total = np.zeros((n, 3))
    for idx, factor in corners(a, shape):
        q = factor*weight.ravel(); mass += np.bincount(idx, weights=q, minlength=n)
        for c in range(3): total[:, c] += np.bincount(idx, weights=q*delta[..., c].ravel(), minlength=n)
    mass = mass.reshape(shape); total = total.reshape(*shape, 3)
    for _ in range(2):
        nm, nt = mass*.5, total*.5
        for axis in (1, 2):
            for direction in [-1, 1]:
                ix = np.clip(np.arange(shape[axis])+direction, 0, shape[axis]-1)
                nm += np.take(mass, ix, axis=axis)/8
                nt += np.take(total, ix, axis=axis)/8
        mass, total = nm, nt
    return np.exp2(total/(mass[..., None]+.3)).astype(np.float32)


def apply_light(fields, a, highlights, shadows, color, top=0, full_shape=None):
    if fields is None: return a
    gain = .2*sample(fields[0], a, top, full_shape)+.8*sample(fields[1], a, top, full_shape)
    target = a*gain; ya, yb = a @ LUMA, target @ LUMA
    good = (ya > 1e-6)&(yb > 1e-6)
    ev = np.clip(np.log2(np.maximum(yb, 1e-6)/np.maximum(ya, 1e-6)), -2, 2)
    scale = np.exp2(ev*np.where(ev >= 0, highlights, shadows))
    correction = target*(ya/np.maximum(yb, 1e-6))[..., None]-a
    return np.where(good[..., None], (a+correction*color)*scale[..., None], a).astype(np.float32)
