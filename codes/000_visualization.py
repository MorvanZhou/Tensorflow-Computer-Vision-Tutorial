import numpy as np
import matplotlib.pyplot as plt

f = np.load('../mnist.npz')
image, label = f['x_train'][7], f['y_train'][7]


def show_conv():
    filter = np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]])
    plt.figure(0, figsize=(9, 5))
    ax1 = plt.subplot(121)
    ax1.imshow(image, cmap='gray')
    plt.xticks(())
    plt.yticks(())
    ax2 = plt.subplot(122)
    plt.ion()
    texts = []
    feature_map = np.zeros((26, 26))
    for i in range(26):
        for j in range(26):

            if texts:
                fm.remove()
            for n in range(3):
                for m in range(3):
                    if len(texts) != 9:
                        texts.append(ax1.text(j+m, i+n, filter[n, m], color='w', size=8, ha='center', va='center',))
                    else:
                        texts[n*3+m].set_position((j+m, i+n))

            feature_map[i, j] = np.sum(filter * image[i:i+3, j:j+3])
            fm = ax2.imshow(feature_map, cmap='gray', vmax=255*3, vmin=-255*3)
            plt.xticks(())
            plt.yticks(())
            plt.pause(0.001)

    plt.ioff()
    plt.show()


def show_result():
    filters = [
        np.array([
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]]),
        np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]]),
        np.array([
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1]]),
        np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]])
    ]

    plt.figure(0)
    plt.title('Original image')
    plt.imshow(image, cmap='gray')
    plt.xticks(())
    plt.yticks(())

    plt.figure(1)
    for n in range(4):
        feature_map = np.zeros((26, 26))

        for i in range(26):
            for j in range(26):
                feature_map[i, j] = np.sum(image[i:i + 3, j:j + 3] * filters[n])

        plt.subplot(3, 4, 1 + n)
        plt.title('Filter%i' % n)
        plt.imshow(filters[n], cmap='gray')
        plt.xticks(())
        plt.yticks(())

        plt.subplot(3, 4, 5 + n)
        plt.title('Conv%i' % n)
        plt.imshow(feature_map, cmap='gray')
        plt.xticks(())
        plt.yticks(())

        plt.subplot(3, 4, 9 + n)
        plt.title('ReLU%i' % n)
        feature_map = np.maximum(0, feature_map)
        plt.imshow(feature_map, cmap='gray')
        plt.xticks(())
        plt.yticks(())

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_conv()
    show_result()