from matplotlib import pyplot as plt


def plot_filters(W, n_width=10, n_height=10, shape=None, title=None):
    plt.figure(figsize=(12, 12))
    for i in xrange(n_height * n_width):
        filters = W[:, i]
        if shape is not None:
            filters = filters.reshape(shape)
        plt.subplot(n_height, n_width, i + 1)
        plt.imshow(filters, cmap=plt.cm.gray, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    if title:
        plt.suptitle(title, fontsize=24)
