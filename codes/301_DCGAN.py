"""
A simple implementation of DCGAN that works on MNIST dataset.
[DCGAN](https://arxiv.org/pdf/1511.06434.pdfï%C2%BC‰)
Learn more, visit my tutorial site: [莫烦Python](https://morvanzhou.github.io)
Dependencies:
tensorflow=1.8.0
numpy=1.14.3
"""
# !pip install imageio

import os, time, itertools, imageio, pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# reset graph
tf.reset_default_graph()

# definition of leaky relu activation
def lrelu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)

# Note: The structures of generator and discriminator are almost symmetric.

# Generator Structure
# Input: random vector of size 100, Output: image to be inputed in discriminator
def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):

        # 1st deconv layer
        conv1 = tf.layers.conv2d_transpose(x, 1024, [4, 4], strides=(1, 1), padding='valid') # ouput size: [batch_size, 4, 4, 1024]
        lrelu1 = lrelu(tf.layers.batch_normalization(conv1, training=isTrain), 0.2)

        # 2nd deconv layer
        conv2 = tf.layers.conv2d_transpose(lrelu1, 512, [4, 4], strides=(2, 2), padding='same') # ouput size: [batch_size, 8, 8, 512]
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd deconv layer
        conv3 = tf.layers.conv2d_transpose(lrelu2, 256, [4, 4], strides=(2, 2), padding='same') # ouput size: [batch_size, 16, 16, 256]
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th deconv layer
        conv4 = tf.layers.conv2d_transpose(lrelu3, 128, [4, 4], strides=(2, 2), padding='same') # ouput size: [batch_size, 32, 32, 128]
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output deconv layer
        conv5 = tf.layers.conv2d_transpose(lrelu4, 1, [4, 4], strides=(2, 2), padding='same') # ouput size: [batch_size, 64, 64, 1]
        o = tf.nn.tanh(conv5)

        return o

# Discriminator Structure
# Input: real data / output of generator, Output: a probability of the input image that is similar to real data
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 1st conv layer
        conv1 = tf.layers.conv2d(x, 128, [4, 4], strides=(2, 2), padding='same') # ouput size: [batch_size, 32, 32, 128]
        lrelu1 = lrelu(conv1, 0.2)

        # 2nd conv layer
        conv2 = tf.layers.conv2d(lrelu1, 256, [4, 4], strides=(2, 2), padding='same') # ouput size: [batch_size, 16, 16, 256]
        lrelu2 = lrelu(tf.layers.batch_normalization(conv2, training=isTrain), 0.2)

        # 3rd conv layer
        conv3 = tf.layers.conv2d(lrelu2, 512, [4, 4], strides=(2, 2), padding='same') # ouput size: [batch_size, 8, 8, 512]
        lrelu3 = lrelu(tf.layers.batch_normalization(conv3, training=isTrain), 0.2)

        # 4th conv layer
        conv4 = tf.layers.conv2d(lrelu3, 1024, [4, 4], strides=(2, 2), padding='same') # ouput size: [batch_size, 4, 4, 1024]
        lrelu4 = lrelu(tf.layers.batch_normalization(conv4, training=isTrain), 0.2)

        # output conv layer
        conv5 = tf.layers.conv2d(lrelu4, 1, [4, 4], strides=(1, 1), padding='valid') # ouput size: [batch_size, 1, 1, 1]

        # to get the probability
        o = tf.nn.sigmoid(conv5)

        return o, conv5

fixed_z = np.random.normal(0, 1, (25, 1, 1, 100))

def show_result(num_epoch, show = False, save = False, path = 'result.png'):
    test_images = sess.run(G_z, {z: fixed_z, isTrain: False})
    size_figure_grid = 5
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(size_figure_grid*size_figure_grid):
        i = k // size_figure_grid
        j = k % size_figure_grid
        ax[i, j].cla()
        ax[i, j].imshow(np.reshape(test_images[k], (64, 64)), cmap='gray')

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()

# training parameters
batch_size = 100
lr = 0.0002
n_epochs = 20

# load MNIST
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

# input placeholders
x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1)) # shape: (batch_size, width, height, channel)
z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
isTrain = tf.placeholder(dtype=tf.bool)

# generator
G_z = generator(z, isTrain)

# discriminator
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

# losses
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1])))

# trainable variables
T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]

# optimizers
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)


    
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# resize and normalization real input dataset as mentioned in DCGAN paper.
train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
train_set = (train_set - 0.5) / 0.5  # scale to [-1 ~ 1] (It's the range of tanh activation) 

# folders to save results
root = 'results/'
model = 'dcgan_'
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(root + 'fixed'):
    os.mkdir(root + 'fixed')

n_batches = mnist.train.num_examples // batch_size

for epoch in range(30):
    
    for iter in range(n_batches):
        d_losses = []
        g_losses = []
        
        # step of discriminator
        x_ = train_set[iter*batch_size:(iter+1)*batch_size]
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_d_, _ = sess.run([D_loss, D_optim], {x: x_, z: z_, isTrain: True})
        d_losses.append(loss_d_)
  
        # step of generator
        z_ = np.random.normal(0, 1, (batch_size, 1, 1, 100))
        loss_g_, _ = sess.run([G_loss, G_optim], {z: z_, x: x_, isTrain: True})
        g_losses.append(loss_g_)
    
    print('[%d/%d] - loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), n_epochs, np.mean(d_losses), np.mean(g_losses) ))
    fixed_p = root + 'fixed/' + model + str(epoch + 1) + '.png'
    show_result((epoch + 1), show=True, save=True, path=fixed_p)


print("Start saving result pngs and gif...")

images = []
for e in range(n_epochs):
    img_name = root + 'fixed/' + model + str(e + 1) + '.png'
    images.append(imageio.imread(img_name))
imageio.mimsave(root + model + 'result.gif', images, fps=5)


sess.close()