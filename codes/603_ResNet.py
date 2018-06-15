"""
A simple implementation of ResNet that works on MNIST dataset.
[ResNet](https://arxiv.org/abs/1512.03385)

Learn more, visit my tutorial site: [莫烦Python](https://morvanzhou.github.io)

Dependencies:
tensorflow=1.8.0
numpy=1.14.3
"""

import numpy as np
import tensorflow as tf

BATCH_SIZE = 64
LR = 0.001              # learning rate

# process mnist data
f = np.load('../mnist.npz')
train_x, train_y = f['x_train'], f['y_train']
test_x, test_y = f['x_test'][:2000], f['y_test'][:2000]
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_x, train_y)).shuffle(1000).repeat(5).batch(BATCH_SIZE)
iterator = train_dataset.make_initializable_iterator()
next_batch = iterator.get_next()

tf_x = tf.placeholder(tf.float32, [None, 28, 28], name='x')/255.*2.-1.  # normalize to (-1, 1)
image = tf.reshape(tf_x, [-1, 28, 28, 1], name='img_x')                 # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, ], name='y')                     # input y


# network structure
def block(x, n_out, strides, n_bottleneck, scope):
    with tf.variable_scope(scope):
        # depth can change in here for projection
        out = bottleneck(x, n_out, scope="bottleneck1", strides=strides,)
        for i in range(1, n_bottleneck):
            out = bottleneck(out, n_out, scope="bottleneck%i" % (1+i),)
    return out


def bottleneck(x, n_out, scope, strides=1,):
    n_in = x.get_shape()[-1]
    with tf.variable_scope(scope):
        b = tf.layers.conv2d(x, n_in, 1, strides=strides, padding='same', activation=tf.nn.relu, name="conv1")
        b = tf.layers.conv2d(b, n_in, 3, strides=1, padding='same', activation=tf.nn.relu, name="conv2")
        b = tf.layers.conv2d(b, n_out, 1, strides=1, padding='same', name="conv3")

        if n_in != n_out:      # projection
            shortcut = tf.layers.conv2d(x, n_out, 1, strides, name="projection")
        else:
            shortcut = x           # identical mapping
        out = tf.nn.relu(shortcut + b)
    return out


with tf.variable_scope('ResNet'):
    net = tf.layers.conv2d(                 # [batch, 28, 28, 1]
        inputs=image,
        filters=16,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        name="conv1")                       # -> [batch, 28, 28, 16]
    net = tf.layers.max_pooling2d(
        inputs=net,
        pool_size=2,
        strides=2,
        name="maxpool1")                    # -> [batch, 14, 14, 16]
    net = block(net, 128, 1, 2, scope="block1")                     # -> [batch, 14, 14, 128]
    net = block(net, 768, 2, 1, scope="block2")                     # -> [batch, 7, 7, 768]
    net = tf.layers.average_pooling2d(net, 7, 1, name="avgpool")    # -> [batch, 1, 1, 768]
    net = tf.layers.flatten(net)                                    # -> [batch, 768]
    logits = tf.layers.dense(net, 10, name='fc4')                   # -> [batch, n_classes]

loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=logits)  # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf_y, predictions=tf.argmax(logits, axis=1),)[1]

sess = tf.Session()
sess.run(tf.group(      # initialize var in graph
    tf.global_variables_initializer(),
    tf.local_variables_initializer(),
    iterator.initializer)
)                       # the local var is for accuracy_op

writer = tf.summary.FileWriter('./log', sess.graph)     # write to file

for step in range(3000):
    b_x, b_y = sess.run(next_batch)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y, })
    if step % 50 == 0:
        accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y,})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)