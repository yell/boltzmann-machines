from matplotlib import pyplot as plt


def plot_matrices(W, n_width=10, n_height=10, shape=None, title=None, imshow_params=None):
    imshow_params = imshow_params or {}
    imshow_params.setdefault('cmap', plt.cm.gray)
    imshow_params.setdefault('interpolation', 'nearest')
    for i in xrange(n_height * n_width):
        matrix = W[i]
        if shape is not None:
            matrix = matrix.reshape(shape)
        ax = plt.subplot(n_height, n_width, i + 1)
        ax.spines['bottom'].set_linewidth(2.)
        ax.spines['top'].set_linewidth(2.)
        ax.spines['left'].set_linewidth(2.)
        ax.spines['right'].set_linewidth(2.)
        plt.tick_params(axis='both', which='both',
                        bottom='off', top='off', left='off', right='off',
                        labelbottom='off', labelleft='off', labelright='off')
        plt.imshow(matrix, **imshow_params)
    if title:
        plt.suptitle(title, fontsize=22)
    plt.subplots_adjust(wspace=0, hspace=0)


def plot_matrices_color(W, n_width=10, n_height=10, shape=None, title=None, imshow_params=None):
    imshow_params = imshow_params or {}
    imshow_params.setdefault('interpolation', 'none')
    for i in xrange(n_height * n_width):
        matrix = W[i]
        if shape is not None:
            matrix = matrix.reshape(shape)
        ax = plt.subplot(n_height, n_width, i + 1)
        ax.spines['bottom'].set_linewidth(2.)
        ax.spines['top'].set_linewidth(2.)
        ax.spines['left'].set_linewidth(2.)
        ax.spines['right'].set_linewidth(2.)
        plt.tick_params(axis='both', which='both',
                        bottom='off', top='off', left='off', right='off',
                        labelbottom='off', labelleft='off', labelright='off')
        plt.imshow(matrix.astype('uint8'), **imshow_params)
    if title:
        plt.suptitle(title, fontsize=22)
    plt.subplots_adjust(wspace=0, hspace=0)
