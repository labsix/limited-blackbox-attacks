from matplotlib import pyplot as plt
from tools.utils import *
import os
import numpy as np
import scipy

def log_output(sess, eval_logits, eval_preds, x, adv, initial_img, \
        target_class, out_dir, orig_class, num_queries):
    """
    Evaluate and save a bunch of metadata about the optimization trajectory and
    the final adversarial examples generated.
    """
    eval_logits_, eval_preds_ = sess.run([eval_logits, eval_preds], {x: adv})
    eval_logits_orig_, eval_preds_orig_ = sess.run([eval_logits, eval_preds], {x: initial_img})
    eval_dir = os.path.join(out_dir, 'eval')
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    with open(os.path.join(eval_dir, 'eval.txt'), 'w') as fout:
        fout.write('true %d\n' % orig_class)
        fout.write('target %d\n' % target_class)
        fout.write('queries %d\n' % num_queries)
    scipy.misc.imsave(os.path.join(eval_dir, 'original.png'), initial_img)
    np.save(os.path.join(eval_dir, 'original.npy'), initial_img)
    scipy.misc.imsave(os.path.join(eval_dir, 'adversarial.png'), adv)
    np.save(os.path.join(eval_dir, 'adversarial.npy'), adv)
    with open(os.path.join(eval_dir, 'sample.txt'), 'w') as fout:
        fout.write('orig_pred %d\n' % eval_preds_orig_[0])
        orig_p = softmax(eval_logits_orig_[0])
        fout.write('orig_conf %.5f\n' % np.max(orig_p))
        fout.write('orig_true_conf %.5f\n' % orig_p[orig_class])
        fout.write('orig_adv_conf %.5f\n' % orig_p[target_class])
        fout.write('adv_pred %d\n' % eval_preds_[0])
        adv_p = softmax(eval_logits_[0])
        fout.write('adv_conf %.5f\n' % np.max(adv_p))
        fout.write('adv_true_conf %.5f\n' % adv_p[orig_class])
        fout.write('adv_adv_conf %.5f\n' % adv_p[target_class])


def render_frame(sess, image, save_index, render_logits, render_feed, out_dir):
    # actually draw the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
    # image
    ax1.imshow(image)
    fig.sca(ax1)
    plt.xticks([])
    plt.yticks([])
    # classifications
    probs = softmax(sess.run(render_logits, {render_feed: image})[0])
    topk = probs.argsort()[-5:][::-1]
    topprobs = probs[topk]
    barlist = ax2.bar(range(5), topprobs)
    for i, v in enumerate(topk):
        if v == orig_class:
            barlist[i].set_color('g')
        if v == target_class:
            barlist[i].set_color('r')
    plt.sca(ax2)
    plt.ylim([0, 1.1])
    plt.xticks(range(5), [label_to_name(i)[:15] for i in topk], rotation='vertical')
    fig.subplots_adjust(bottom=0.2)
    path = os.path.join(out_dir, 'frame%06d.png' % save_index)
    if os.path.exists(path):
        os.remove(path)
    plt.savefig(path)
    plt.close()
