import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def plot_matrices(X, n_width=10, n_height=10, shape=None, normalize=True, title=None,
                  title_params=None, imshow_params=None):
    """Draw grid of matrices represented by rows of `X`.

    Returns
    -------
    im : matplotlib.image.AxesImage
    X_reshaped : (`n_height` * `shape`[0], `n_width` * `shape`[1], `shape`[2]) np.ndarray
    """
    # check params
    X = np.asarray(X)

    if shape is None:
        shape = X.shape[1:]

    title_params = title_params or {}
    title_params.setdefault('fontsize', 22)

    imshow_params = imshow_params or {}
    imshow_params.setdefault('interpolation', 'nearest')

    # reshape `X`
    Y = X[:(n_width * n_height), ...].copy()
    if len(shape) == 2:
        shape = (shape[0], shape[1], 1)
    Y = Y.reshape(-1, *shape)
    Z = np.zeros((n_height * shape[0], n_width * shape[1], shape[2]))

    for i in xrange(n_height):
        for j in xrange(n_width):
            ind_Y = n_height * i + j
            if ind_Y < len(Y):
                Y_i = Y[ind_Y, ...]
                if normalize:
                    Y_i -= Y_i.min()
                    Y_i /= max(Y_i.ptp(), 1e-5)
                    Y_i /= Y_i.max()
                Z[i * shape[0]:(i + 1) * shape[0],
                  j * shape[1]:(j + 1) * shape[1], ...] = Y_i
    if Z.shape[2] == 1:
        Z = Z[:, :, 0]

    # plot reshaped `X`
    im = plt.imshow(Z, **imshow_params)
    if title:
        im.axes.set_title(title, **title_params)
    im.axes.tick_params(axis='both', which='both',
                        bottom='off', top='off', left='off', right='off',
                        labelbottom='off', labelleft='off', labelright='off')
    return im, Z


def gif_matrices(matrices, im, fig, fname='samples.gif', title_func=None,
                 title_params=None, anim_params=None, save_params=None):
    """Animate `matrices`.

    Parameters
    ----------
    matrices : [np.ndarray]
        list of matrices to animate

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    if title_func is None:
        title_func = lambda i: str(i)

    title_params = title_params or {}
    title_params.setdefault('fontsize', 18)

    save_params = save_params or {}
    save_params.setdefault('dpi', 80)
    save_params.setdefault('writer', 'imagemagick')

    anim_params = anim_params or {}
    anim_params.setdefault('interval', 200)
    anim_params.setdefault('blit', True)

    def init():
        im.set_array([[]])
        return im,

    def animate(i):
        im.set_array(matrices[i])
        title = title_func(i)
        im.axes.set_title(title, **title_params)
        return im,

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(matrices), **anim_params)
    anim.save(fname, **save_params)
    return anim


def plot_confusion_matrix(C, labels=None, labels_fontsize=None, **heatmap_params):
    # default params
    labels = labels or range(C.shape[0])
    labels_fontsize = labels_fontsize or 13
    annot_fontsize = 14
    xy_label_fontsize = 21

    # set default params where possible
    if not 'annot' in heatmap_params:
        heatmap_params['annot'] = True
    if not 'fmt' in heatmap_params:
        heatmap_params['fmt'] = 'd' if C.dtype is np.dtype('int') else '.3f'
    if not 'annot_kws' in heatmap_params:
        heatmap_params['annot_kws'] = {'size': annot_fontsize}
    elif not 'size' in heatmap_params['annot_kws']:
        heatmap_params['annot_kws']['size'] = annot_fontsize
    if not 'xticklabels' in heatmap_params:
        heatmap_params['xticklabels'] = labels
    if not 'yticklabels' in heatmap_params:
        heatmap_params['yticklabels'] = labels

    # plot the stuff
    with plt.rc_context(rc={'xtick.labelsize': labels_fontsize,
                            'ytick.labelsize': labels_fontsize}):
        ax = sns.heatmap(C, **heatmap_params)
        plt.xlabel('predicted', fontsize=xy_label_fontsize)
        plt.ylabel('actual', fontsize=xy_label_fontsize)
        return ax
