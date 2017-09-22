import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def plot_matrices(W, n_width=10, n_height=10, shape=None, title=None, imshow_params=None):
    """Draw grid of matrices represented by rows of `W`."""
    imshow_params = imshow_params or {}
    imshow_params.setdefault('interpolation', 'nearest')
    for i in xrange(n_height * n_width):
        matrix = W[i]
        if shape is not None:
            matrix = matrix.reshape(shape)
        ax = plt.subplot(n_height, n_width, i + 1)
        for d in ('bottom', 'top', 'left', 'right'):
            ax.spines[d].set_linewidth(2.)
        plt.tick_params(axis='both', which='both',
                        bottom='off', top='off', left='off', right='off',
                        labelbottom='off', labelleft='off', labelright='off')
        plt.imshow(matrix, **imshow_params)
    if title:
        plt.suptitle(title, fontsize=22)
    plt.subplots_adjust(wspace=0, hspace=0)

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
