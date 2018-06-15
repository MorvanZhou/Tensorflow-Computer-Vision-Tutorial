"""
A simple implementation of LeNet that works on MNIST dataset.
[LeNet](http://www.dengfanxin.cn/wp-content/uploads/2016/03/1998Lecun.pdf)

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
with tf.variable_scope('LeNet'):
    net = tf.layers.conv2d(                 # [batch, 28, 28, 1]
        inputs=image,
        filters=6,
        kernel_size=5,
        strides=1,
        padding='same',
        name="conv1")                       # -> [batch, 28, 28, 6]
    net = tf.layers.max_pooling2d(
        inputs=net,
        pool_size=2,
        strides=2,
        name="maxpool1")                                        # -> [batch, 14, 14, 6]
    net = tf.layers.conv2d(net, 16, 5, 1, name="conv2")         # -> [batch, 14, 14, 16]
    net = tf.layers.max_pooling2d(net, 2, 2, name="maxpool2")   # -> [batch, 7, 7, 16]
    net = tf.layers.flatten(net, name='flat')                   # -> [batch, 7*7*16=784]
    logits = tf.layers.dense(net, 10, name='fc4')               # -> [batch, n_classes]

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
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y})
        print('Step:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)