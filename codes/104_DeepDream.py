"""
The introduction of DeepDream on [Google's blog post](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html)
This code implementation is based on [tensorflow deepdream tutorial](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream)

Learn more, visit my tutorial site: [莫烦Python](https://morvanzhou.github.io)

Dependencies:
tensorflow=1.8.0
PIL=5.1.0
requests=2.18.4
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import requests, os, zipfile

MODEL_PATH = '../models/tensorflow_inception_graph.pb'


def maybe_download(model_path):
    if not os.path.isfile(model_path):
        print('downloading...')
        with open("../inception5h.zip", 'wb') as f:
            f.write(requests.get("https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip").content)
        os.makedirs('../models', exist_ok=True)
        with zipfile.ZipFile("../inception5h.zip", 'r') as zip_ref:
            zip_ref.extractall('../models/')
        os.remove('../inception5h.zip')
        os.remove('../models/imagenet_comp_graph_label_strings.txt')
        os.remove('../models/LICENSE')
        print('download to ' + model_path)
    return model_path


# load model to the graph
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
model_path = maybe_download(MODEL_PATH)
with tf.gfile.FastGFile(model_path, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# add input to the graph
tf_input = tf.placeholder(tf.float32, name="input")
imagenet_mean = 117.0
tf_preprocessed = tf.expand_dims(tf_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {"input": tf_preprocessed})

# find a layer and channel depends on the graph showing in tensorboard
tf.summary.FileWriter('./log', sess.graph)


def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap


# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in range(0, max(h-sz//2, sz),sz):
        for x in range(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {tf_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)


def render_deepdream(tf_obj, img0, save_path, iter_n=50, step=1.5, octave_n=4, octave_scale=1.4):
    # backprop from this tf_obj
    t_score = tf.reduce_mean(tf_obj)                # defining the optimization objective
    t_grad = tf.gradients(t_score, tf_input)[0]     # the impact on the input layer

    # split the image into a number of octaves
    img = img0
    octaves = []
    for i in range(octave_n - 1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw) / octave_scale))
        hi = img - resize(lo, hw)
        img = lo
        octaves.append(hi)

    # generate details octave by octave
    print('dreaming of image: ' + save_path)
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            img += g * (step / (np.abs(g).mean() + 1e-7))
    Image.fromarray(img.clip(0, 255).astype(np.uint8)).save(save_path)


# picking a layer and channel from tensorboard to visualize
image_path = "../example_images/morvan.jpg"
layer = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 123
os.makedirs('../results', exist_ok=True)
output_path = '../results/' + image_path.split('/')[-1].split('.')[0] + '_' + layer + '_%i.jpeg' % channel
layer_channel = graph.get_tensor_by_name("import/%s:0" % layer)[:, :, :, channel]

# test on a noise image
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0
render_deepdream(layer_channel, img_noise, './results/noise_dream.jpeg')

# test on a real image
img = Image.open(image_path)
img.load()
render_deepdream(layer_channel, np.asarray(img, dtype=np.float32), output_path)


