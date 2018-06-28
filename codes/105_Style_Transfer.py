"""
https://harishnarayanan.org/writing/artistic-style-transfer/
https://github.com/hnarayanan/artistic-style-transfer/blob/master/notebooks/6_Artistic_style_transfer_with_a_repurposed_VGG_Net_16.ipynb

https://harishnarayanan.org/writing/artistic-style-transfer/
"""

import tensorflow as tf
import numpy as np
from PIL import Image
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt

# image and model path
CONTENT_PATH = '../example_images/morvan3.jpg'
STYLE_PATH = '../example_images/style4.jpg'
VGG_PATH = '../models/vgg16.npy'

# weight for loss (content loss, style loss and total variation loss)
W_CONTENT = 0.001
W_STYLE = W_CONTENT * 1e2
W_VARIATION = 1.
HEIGHT, WIDTH = 400, 400    # output image height and width
N_ITER = 6                  # styling how many times?


class StyleTransfer:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy, w_content, w_style, w_variation, height, width):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        self.height, self.width = height, width

        # network input (combined images)
        self.tf_content = tf.placeholder(tf.float32, [1, height, width, 3])
        self.tf_style = tf.placeholder(tf.float32, [1, height, width, 3])
        self.tf_styled = tf.placeholder(tf.float32, [1, height, width, 3])
        concat_image = tf.concat((self.tf_content, self.tf_style, self.tf_styled), axis=0)    # combined input

        # convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=concat_image)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG conv layers
        self.conv1_1 = self._conv_layer(bgr, "conv1_1")
        self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self._max_pool(self.conv1_2, 'pool1')
        self.conv2_1 = self._conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self._conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self._max_pool(self.conv2_2, 'pool2')
        self.conv3_1 = self._conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self._conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self._conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self._max_pool(self.conv3_3, 'pool3')
        self.conv4_1 = self._conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self._conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self._conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self._max_pool(self.conv4_3, 'pool4')
        self.conv5_1 = self._conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self._conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self._conv_layer(self.conv5_2, "conv5_3")

        # we don't need fully connected layers for style transfer

        with tf.variable_scope('content_loss'):     # compute content loss
            content_feature_maps = self.conv2_2[0]
            styled_feature_maps = self.conv2_2[2]
            loss = w_content * tf.reduce_sum(tf.square(content_feature_maps-styled_feature_maps))

        with tf.variable_scope('style_loss'):       # compute style loss
            conv_layers = [self.conv1_2, self.conv2_2, self.conv3_3, self.conv4_3, self.conv5_3]
            for conv_layer in conv_layers:
                style_feature_maps = conv_layer[1]
                styled_feature_maps = conv_layer[2]
                style_loss = (w_style / len(conv_layers)) * self._style_loss(style_feature_maps, styled_feature_maps)
                loss = tf.add(loss, style_loss)     # combine losses

        with tf.variable_scope('variation_loss'):   # total variation loss, reduce noise
            a = tf.square(self.tf_styled[:, :height - 1, :width - 1, :] - self.tf_styled[:, 1:, :width - 1, :])
            b = tf.square(self.tf_styled[:, :height - 1, :width - 1, :] - self.tf_styled[:, :height - 1, 1:, :])
            variation_loss = w_variation * tf.reduce_sum(tf.pow(a + b, 1.25))
            self.loss = tf.add(loss, variation_loss)

        # styled image's gradient
        self.grads = tf.gradients(loss, self.tf_styled)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('./log', self.sess.graph)

    def styling(self, content_image, style_image, n_iter):
        content = Image.open(content_image).resize((self.width, self.height))
        self.content = np.expand_dims(content, axis=0).astype(np.float32)   # [1, height, width, 3]
        style = Image.open(style_image).resize((self.width, self.height))
        self.style = np.expand_dims(style, axis=0).astype(np.float32)       # [1, height, width, 3]

        x = np.copy(self.content)      # initialize styled image from content
        
        # repeat backpropagating to styled image 
        for i in range(n_iter):
            x, min_val, info = fmin_l_bfgs_b(self._get_loss, x.flatten(), fprime=lambda x: self.flat_grads, maxfun=20)
            x = x.clip(0., 255.)
            print(i, ' loss: ', min_val)

        x = x.reshape((self.height, self.width, 3))
        for i in range(1, 4):
            x[:, :, -i] += self.vgg_mean[i - 1]
        return x, self.content, self.style
    
    def _get_loss(self, x):
        loss, grads = self.sess.run(
            [self.loss, self.grads], feed_dict={
                self.tf_styled: x.reshape((1, self.height, self.width, 3)),
                self.tf_content: self.content,
                self.tf_style: self.style
            })
        self.flat_grads = grads[0].flatten().astype(np.float64)
        return loss

    def _style_loss(self, style_feature, styled_feature):
        def gram_matrix(x):
            num_channels = int(x.get_shape()[-1])
            matrix = tf.reshape(x, shape=[-1, num_channels])
            gram = tf.matmul(tf.transpose(matrix), matrix)
            return gram

        s = gram_matrix(style_feature)
        t = gram_matrix(styled_feature)
        channels = 3
        size = self.width * self.height
        return tf.reduce_sum(tf.square(s - t)) / (4. * (channels ** 2) * (size ** 2))

    def _max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def _conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # in here, CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout


image_filter = StyleTransfer(VGG_PATH, W_CONTENT, W_STYLE, W_VARIATION, HEIGHT, WIDTH,)
image, content_image, style_image = image_filter.styling(CONTENT_PATH, STYLE_PATH, N_ITER)     # style transfer

# save
image = image.clip(0, 255).astype(np.uint8)
save_name = '_'.join([path.split('/')[-1].split('.')[0] for path in [CONTENT_PATH, STYLE_PATH]]) + '.jpeg'
Image.fromarray(image).save('../results/%s' % save_name)    # save result

# plotting
plt.figure(1, figsize=(8, 4))
plt.subplot(131)
plt.imshow(content_image.reshape((HEIGHT, WIDTH, 3)).astype(int))
plt.title('Content')
plt.xticks(());plt.yticks(())
plt.subplot(132)
plt.imshow(style_image.reshape((HEIGHT, WIDTH, 3)).astype(int))
plt.title('Style')
plt.xticks(());plt.yticks(())
plt.subplot(133)
plt.title('styled')
plt.imshow(image)
plt.xticks(());plt.yticks(())
plt.tight_layout()
plt.show()
