from tools import utils
from tools.inception_v3_imagenet import model
import tensorflow as tf
import pickle
import sys

IMAGENET_PATH=""
if __name__=="__main__":
    if IMAGENET_PATH == "":
        raise ValueError("Please open precompute.py and set IMAGENET_PATH")
    s = (299, 299, 3)
    dataset = sys.argv[1]
    last_j = 0
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, s)
    _, preds = model(sess, tf.expand_dims(x, 0))
    label_dict = {}
    for i in range(1,1000):
        print("Looking for %d" % (i,))
        if i in label_dict:
            continue
        for j in range(last_j, 50000):
            im, lab = utils.get_image(j, IMAGENET_PATH)
            if sess.run(preds, {x: im})[0] == lab:
                label_dict[lab] = j
            if lab == i:
                label_dict[i] = j
                break
        last_j = j
    pickle.dump(label_dict, open("tools/data/imagenet.pickle", "wb"))
