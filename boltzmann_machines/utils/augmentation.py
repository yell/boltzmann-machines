import numpy as np
import scipy.ndimage as nd


def shift(x, offset=(0, 0)):
    if len(x.shape) == 3:
        y = np.zeros_like(x)
        for c in range(x.shape[2]):
            y[:, :, c] = shift(x[:, :, c], offset=offset)
        return y
    y = nd.interpolation.shift(x, shift=offset, mode='nearest')
    return y

def horizontal_mirror(x):
    y = np.fliplr(x[:,:,...])
    return y
