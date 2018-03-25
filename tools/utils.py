import tensorflow as tf
import PIL
from tensorflow.python.framework import ops
import numpy as np
import os
import pickle
from matplotlib import pyplot as plt

def image_of_class(y, imagenet_path=None):
    """
    Gets an image of a prespecified class. To save computation time we use a
    presaved dictionary of an index for each class, but an attacker can also
    randomize this, or search for the best starting images.
    """
    im_indices = pickle.load(open("tools/data/imagenet.pickle", "rb"))
    return get_image(im_indices[y], imagenet_path)[0].copy()

def pseudorandom_target(index, total_indices, true_class):
    rng = np.random.RandomState(index)
    target = true_class
    while target == true_class:
        target = rng.randint(0, total_indices)
    return target

def pseudorandom_target_image(orig_index, total_indices):
    rng = np.random.RandomState(orig_index)
    target_img_index = orig_index
    while target_img_index == orig_index:
        target_img_index = rng.randint(0, total_indices)
    return target_img_index

# get center crop
def load_image(path):
    image = PIL.Image.open(path)
    if image.height > image.width:
        height_off = int((image.height - image.width)/2)
        image = image.crop((0, height_off, image.width, height_off+image.width))
    elif image.width > image.height:
        width_off = int((image.width - image.height)/2)
        image = image.crop((width_off, 0, width_off+image.height, image.height))
    image = image.resize((299, 299))
    img = np.asarray(image).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.repeat(img[:,:,np.newaxis], repeats=3, axis=2)
    if img.shape[2] == 4:
        # alpha channel
        img = img[:,:,:3]
    return img

def get_image(index, imagenet_path=None):
    data_path = os.path.join(imagenet_path, 'val')
    image_paths = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)])
    assert len(image_paths) == 50000
    labels_path = os.path.join(imagenet_path, 'val.txt')
    with open(labels_path) as labels_file:
        labels = [i.split(' ') for i in labels_file.read().strip().split('\n')]
        labels = {os.path.basename(i[0]): int(i[1]) for i in labels}
    def get(index):
        path = image_paths[index]
        x = load_image(path)
        y = labels[os.path.basename(path)]
        return x, y
    return get(index)

def one_hot(index, total):
    arr = np.zeros((total))
    arr[index] = 1.0
    return arr

def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
            if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def int_shape(tensor):
    return list(map(int, tensor.get_shape()))

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def hms(seconds):
    seconds = int(seconds)
    hours = (seconds // (60 * 60))
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    if hours > 0:
        return '%d hrs %d min' % (hours, minutes)
    elif minutes > 0:
        return '%d min %d sec' % (minutes, seconds)
    else:
        return '%d sec' % seconds

_py_func_id = 0
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    global _py_func_id

    rnd_name = 'PyFuncGrad' + '%08d' % _py_func_id
    _py_func_id += 1

    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

def grad_clip_by_norm(x, clip_norm=1, name=None):
    if isinstance(clip_norm, int):
        clip_norm = float(clip_norm)
    with ops.name_scope(name, "grad_clip_by_norm", [x, clip_norm]) as name:
        identity, = py_func(
            lambda t,
            _: t,
            [x, clip_norm],
            [tf.float32],
            name=name,
            grad=_grad_clip_by_norm_grad,
            stateful=False
        )
        identity.set_shape(x.get_shape())
        return identity

# Actual gradient:
def _grad_clip_by_norm_grad(op, grad):
    _, norm = op.inputs
    return (tf.clip_by_norm(grad, norm), None)

def grad_clip_by_value(x, clip_magnitude=1, name=None):
    if isinstance(clip_magnitude, int):
        clip_magnitude = float(clip_magnitude)
    with ops.name_scope(name, "grad_clip_by_value", [x, clip_magnitude]) as name:
        identity, = py_func(
            lambda t,
            _: t,
            [x, clip_magnitude],
            [tf.float32],
            name=name,
            grad=_grad_clip_by_value_grad,
            stateful=False
        )
        identity.set_shape(x.get_shape())
        return identity

# Actual gradient:
def _grad_clip_by_value_grad(op, grad):
    _, mag = op.inputs
    return (tf.clip_by_value(grad, -mag, mag), None)
