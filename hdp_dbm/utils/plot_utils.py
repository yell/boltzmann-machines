from matplotlib import pyplot as plt


def plot_matrices(W, n_width=10, n_height=10, shape=None, title=None):
    plt.figure(figsize=(n_width + 2, n_height + 2))
    for i in xrange(n_height * n_width):
        matrix = W[i]
        if shape is not None:
            matrix = matrix.reshape(shape)
        plt.subplot(n_height, n_width, i + 1)
        plt.imshow(matrix, cmap=plt.cm.gray, interpolation='nearest')
        plt.axis('off')
    if title:
        plt.suptitle(title, fontsize=24)


def plot_matrices_color(W, n_width=10, n_height=10, shape=None, title=None):
    for i in xrange(n_height * n_width):
        matrix = W[i]
        if shape is not None:
            matrix = matrix.reshape(shape)
        plt.subplot(n_height, n_width, i + 1)
        plt.imshow(matrix.astype('uint8'), interpolation='none')
        plt.axis('off')
    if title:
        plt.suptitle(title, fontsize=24)
