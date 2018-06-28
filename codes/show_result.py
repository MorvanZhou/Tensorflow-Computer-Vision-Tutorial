import matplotlib.pyplot as plt
from PIL import Image


def style_transfer():
    # plotting
    content = Image.open('../example_images/morvan3.jpg').resize((400, 400))
    plt.figure(1, figsize=(4, 6))
    for i in range(4):
        for j in range(3):
            plt.subplot(4, 3, 3*i+j+1)
            if j == 0:
                plt.imshow(content)
                if i == 0:
                    plt.title('Content')
            elif j == 1:
                style = Image.open('../example_images/style%i.jpg' % (i+1)).resize((400, 400))
                plt.imshow(style)
                if i == 0:
                    plt.title('Style')
            else:
                styled = Image.open('../results/morvan3_style%i.jpeg' % (i+1)).resize((400, 400))
                plt.imshow(styled)
                if i == 0:
                    plt.title('Styled')
            plt.xticks(());plt.yticks(())
    plt.tight_layout()
    plt.savefig('../results/sum_style_transfer.png', dpi=500)
    # plt.show()


if __name__ == '__main__':
    style_transfer()