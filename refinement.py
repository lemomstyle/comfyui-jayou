import numpy as np
from .fields import smooth_grid
from .common import interrupted


def color_corners(a, size):
    p = np.clip(a, 0, 1).reshape(-1, 3)*(size-1)
    base = np.floor(p).astype(np.int32); f = p-base
    for z in range(2):
        for y in range(2):
            for x in range(2):
                idx = ((np.minimum(size-1, base[:, 2]+z)*size+np.minimum(size-1, base[:, 1]+y))*size
                       +np.minimum(size-1, base[:, 0]+x))
                q = (f[:, 0] if x else 1-f[:, 0])*(f[:, 1] if y else 1-f[:, 1])*(f[:, 2] if z else 1-f[:, 2])
                yield idx, q


def fit_lut(a, b, weights, size=9):
    n = size**3; mass = np.zeros(n); total = np.zeros((n, 3))
    weight = weights.ravel().copy(); weight[np.arange(weight.size) % 7 == 0] = 0
    residual = np.clip(b-a, -.25, .25).reshape(-1, 3)
    for idx, factor in color_corners(a, size):
        q = factor*weight; mass += np.bincount(idx, weights=q, minlength=n)
        for c in range(3): total[:, c] += np.bincount(idx, weights=q*residual[:, c], minlength=n)
    interrupted()
    values = (total/(mass[:, None]+2)).reshape(size, size, size, 3)
    return smooth_grid(values, mass.reshape(size, size, size), 12)


def apply_lut(lut, a):
    result = np.zeros((a.shape[0]*a.shape[1], 3), np.float32)
    flat = lut.reshape(-1, 3)
    for idx, q in color_corners(a, lut.shape[0]): result += flat[idx]*q[:, None]
    return a+result.reshape(a.shape)
