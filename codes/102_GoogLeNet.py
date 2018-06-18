"""
A simple implementation of GoogLeNet that works on MNIST dataset.
[GoogLeNet](https://arxiv.org/abs/1409.4842)

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
f = np.load('./mnist.npz')
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
def inception(x, p1, p2, p3, p4, scope):
    p1f11 = p1
    p2f11, p2f33 = p2
    p3f11, p3f55 = p3
    p4f11 = p4
    with tf.variable_scope(scope):
        path1 = tf.layers.conv2d(x, filters=p1f11, kernel_size=1, activation=tf.nn.relu, name='p1f11')

        path2 = tf.layers.conv2d(x, p2f11, 1, activation=tf.nn.relu, name='p2f11')
        path2 = tf.layers.conv2d(path2, p2f33, 3, padding='same', activation=tf.nn.relu, name='p2f33')

        path3 = tf.layers.conv2d(x, p3f11, 1, activation=tf.nn.relu, name='p3f11')
        path3 = tf.layers.conv2d(path3, p3f55, 5, padding='same', activation=tf.nn.relu, name='p3f55')

        path4 = tf.layers.max_pooling2d(x, pool_size=3, strides=1, padding='same', name='p4p33')
        path4 = tf.layers.conv2d(path4, p4f11, 1, activation=tf.nn.relu, name='p4f11')

        out = tf.concat((path1, path2, path3, path4), axis=-1, name='path_cat')
    return out


with tf.variable_scope('GoogLeNet'):
    net = tf.layers.conv2d(                 # [batch, 28, 28, 1]
        inputs=image,
        filters=12,
        kernel_size=5,
        strides=1,
        padding='same',
        name="conv1")                       # -> [batch, 28, 28, 12]
    net = tf.layers.max_pooling2d(net, 2, 2, name="maxpool1")                   # -> [batch, 14, 14, 12]
    net = inception(net, p1=64, p2=(6, 64), p3=(6, 32), p4=32, scope='incpt1')  # -> [batch, 14, 14, 64+64+32+32=192]
    net = tf.layers.max_pooling2d(net, 3, 2, padding='same', name="maxpool1")   # -> [batch, 7, 7, 192]
    net = inception(net, p1=256, p2=(32, 256), p3=(32, 128), p4=128, scope='incpt2')  # -> [batch, 7, 7, 768]
    net = tf.layers.average_pooling2d(net, 7, 1, name="avgpool")                # -> [batch, 1, 1, 768]
    net = tf.layers.flatten(net, name='flat')                                   # -> [batch, 768]
    logits = tf.layers.dense(net, 10, name='fc4')                               # -> [batch, n_classes]

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
        accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y, })
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)