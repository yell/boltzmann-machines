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
