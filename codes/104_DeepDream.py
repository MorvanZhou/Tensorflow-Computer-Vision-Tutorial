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


# picking a layer and channel from tensorboard to visualize
IMAGE_PATH = "../example_images/morvan1.jpg"
LAYER = 'mixed4d_3x3_bottleneck_pre_relu'       # try finding layer name on tensorboard
CHANNEL = 60
MODEL_PATH = '../models/tensorflow_inception_graph.pb'
OUTPUT_DIR = '../results/'


def tf_func(*argtypes):
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
def tf_resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]


class DeepDream(object):
    def __init__(self, model_path):
        # load model to the graph
        self.graph = tf.Graph()
        self.sess = tf.InteractiveSession(graph=self.graph)
        model_path = self.maybe_download(model_path)        # try downloading CNN model

        with tf.gfile.FastGFile(model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # add input to the graph
        self.tf_input = tf.placeholder(tf.float32, name="input")
        imagenet_mean = 117.0
        tf_preprocessed = tf.expand_dims(self.tf_input - imagenet_mean, 0)
        tf.import_graph_def(graph_def, {"input": tf_preprocessed})

        # find a layer and channel depends on the graph showing in tensorboard
        tf.summary.FileWriter('./log', self.sess.graph)
        print('The graph is save to ./log, you can now pick a layer name from tensorboard or select one in below\n')
        for op in self.graph.get_operations():
            if op.type == 'Conv2D' and 'import/' in op.name:
                print(op.name[7:])

        self.resize = tf_func(np.float32, np.int32)(tf_resize)

    @staticmethod
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

    def calc_grad_tiled(self, img, t_grad, tile_size=512):
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
                g = self.sess.run(t_grad, {self.tf_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    def dream(self, image_path, layer, channel, iter_n=50, step=1.5, octave_n=4, octave_scale=1.4):
        # load base image
        img = np.asarray(Image.open(image_path), dtype=np.float32)

        # backprop from this layer_channel
        layer_channel = self.graph.get_tensor_by_name("import/%s:0" % layer)[:, :, :, channel]
        t_score = tf.reduce_mean(layer_channel)                 # defining the optimization objective
        t_grad = tf.gradients(t_score, self.tf_input)[0]        # the impact on the input layer

        # split the image into a number of octaves
        octaves = []
        for i in range(octave_n - 1):
            hw = img.shape[:2]
            lo = self.resize(img, np.int32(np.float32(hw) / octave_scale))
            hi = img - self.resize(lo, hw)
            img = lo
            octaves.append(hi)

        # generate details octave by octave
        for octave in range(octave_n):
            print('dreaming... (%i/%i)' % (octave+1, octave_n))
            if octave > 0:
                hi = octaves[-octave]
                img = self.resize(img, hi.shape[:2]) + hi
            for i in range(iter_n):
                g = self.calc_grad_tiled(img, t_grad)
                img += g * (step / (np.abs(g).mean() + 1e-7))
        return img.clip(0, 255).astype(np.uint8)


if __name__ == '__main__':
    # create model and make a sweet dream
    deep_dream = DeepDream(MODEL_PATH)
    img = deep_dream.dream(IMAGE_PATH, LAYER, CHANNEL)

    # save
    save_name = '_'.join([IMAGE_PATH.split('/')[-1].split('.')[0], LAYER, str(CHANNEL)]) + '.jpeg'
    output_path = ''.join([OUTPUT_DIR, save_name])
    Image.fromarray(img).save(output_path)


