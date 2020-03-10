from PIL import Image
import numpy as np
import tensorflow as tf
from tools.utils import *
import json
import pdb
import os
import sys
import time
import scipy.misc
import PIL

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.logging_utils import *

from tools.inception_v3_imagenet import model
from tools.imagenet_labels import label_to_name

IMAGENET_PATH=""
NUM_LABELS=1000
SIZE = 299

def main(args, gpus):
    # INITIAL IMAGE AND CLASS SELECTION
    if args.img_path:
        initial_img = np.asarray(Image.open(args.img_path).resize((SIZE, SIZE)))
        orig_class = args.orig_class
        initial_img = initial_img.astype(np.float32) / 255.0
    else:
        x, y = get_image(args.img_index, IMAGENET_PATH)
        orig_class = y
        initial_img = x

    # PARAMETER SETUP
    if args.target_class is None:
        target_class = pseudorandom_target(args.img_index, NUM_LABELS, orig_class)
        print('chose pseudorandom target class: %d' % target_class)
    else:
        target_class = args.target_class
    batch_size = args.batch_size
    out_dir = args.out_dir
    epsilon = args.epsilon
    lower = np.clip(initial_img - args.epsilon, 0., 1.)
    upper = np.clip(initial_img + args.epsilon, 0., 1.)
    adv = initial_img.copy() if not args.restore else \
            np.clip(np.load(args.restore), lower, upper)
    batch_per_gpu = batch_size // len(gpus)
    log_iters = args.log_iters
    queries_per_iter = args.samples_per_draw
    max_iters = int(np.ceil(args.max_queries // queries_per_iter))
    max_lr = args.max_lr
    # ----- partial info params -----
    k = args.top_k
    goal_epsilon = epsilon
    adv_thresh = args.adv_thresh
    if k > 0:
        if target_class == -1:
            raise ValueError("Partial-information attack is a targeted attack.")
        adv = image_of_class(target_class, IMAGENET_PATH)
        epsilon = args.starting_eps
        delta_epsilon = args.starting_delta_eps
    else:
        k = NUM_LABELS
    # ----- label only params -----
    label_only = args.label_only
    zero_iters = args.zero_iters

    # TARGET CLASS SELECTION
    if target_class < 0:
        one_hot_vec = one_hot(orig_class, NUM_LABELS)
    else:
        one_hot_vec = one_hot(target_class, NUM_LABELS)
    labels = np.repeat(np.expand_dims(one_hot_vec, axis=0),
                       repeats=batch_per_gpu, axis=0)
    is_targeted = 1 if target_class >= 0 else -1

    # SESSION INITIALIZATION
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32, initial_img.shape)
    eval_logits, eval_preds = model(sess, tf.expand_dims(x, 0))
    if target_class >= 0:
        eval_percent_adv = tf.equal(eval_preds[0], tf.constant(target_class, tf.int64))
    else:
        eval_percent_adv = tf.not_equal(eval_preds[0], tf.constant(orig_class, tf.int64))

    # TENSORBOARD SETUP
    empirical_loss = tf.placeholder(dtype=tf.float32, shape=())
    lr_placeholder = tf.placeholder(dtype=tf.float32, shape=())
    loss_vs_queries = tf.summary.scalar('empirical loss vs queries', empirical_loss)
    loss_vs_steps = tf.summary.scalar('empirical loss vs step', empirical_loss)
    lr_vs_queries = tf.summary.scalar('lr vs queries', lr_placeholder)
    lr_vs_steps = tf.summary.scalar('lr vs step', lr_placeholder)
    writer = tf.summary.FileWriter(out_dir, graph=sess.graph)
    log_file = open(os.path.join(out_dir, 'log.txt'), 'w+')
    with open(os.path.join(out_dir, 'args.json'), 'w') as args_file:
        json.dump(args.__dict__, args_file)

    # LOSS FUNCTION
    def standard_loss(eval_points, noise):
        logits, preds = model(sess, eval_points)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        return losses, noise

    def label_only_loss(eval_points, noise):
        noised_eval_points = tf.zeros((batch_per_gpu,))
        tiled_points = tf.tile(tf.expand_dims(eval_points, 0), [zero_iters,1,1,1,1])
        noised_eval_im = tiled_points + \
                tf.random_uniform(tf.shape(tiled_points), minval=-1, \
                maxval=1)*args.label_only_sigma
        logits, preds = model(sess, tf.reshape(noised_eval_im, (-1,) + initial_img.shape))
        vals, inds = tf.nn.top_k(logits, k=k)
        real_inds = tf.reshape(inds, (zero_iters, batch_per_gpu, -1))
        rank_range = tf.range(start=k, limit=0, delta=-1, dtype=tf.float32)
        tiled_rank_range = tf.tile(tf.reshape(rank_range, (1, 1, k)), [zero_iters, batch_per_gpu, 1])
        batches_in = tf.where(tf.equal(real_inds, target_class), 
                tiled_rank_range, tf.zeros(tf.shape(tiled_rank_range)))
        return 1 - tf.reduce_mean(batches_in, [0, 2]), noise

    def partial_info_loss(eval_points, noise):
        logits, preds = model(sess, eval_points)
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        vals, inds = tf.nn.top_k(logits, k=k)
        # inds is batch_size x k
        good_inds = tf.where(tf.equal(inds, tf.constant(target_class))) # returns (# true) x 3
        good_images = good_inds[:,0] # inds of img in batch that worked
        losses = tf.gather(losses, good_images)
        noise = tf.gather(noise, good_images)
        return losses, noise

    # GRADIENT ESTIMATION GRAPH
    grad_estimates = []
    final_losses = []
    loss_fn = label_only_loss if label_only else \
                (partial_info_loss if k < NUM_LABELS else standard_loss)
    for i, device in enumerate(gpus):
        with tf.device(device):
            print('loading on gpu %d of %d' % (i+1, len(gpus)))
            noise_pos = tf.random_normal((batch_per_gpu//2,) + initial_img.shape)
            noise = tf.concat([noise_pos, -noise_pos], axis=0)
            eval_points = x + args.sigma * noise
            losses, noise = loss_fn(eval_points, noise)
        losses_tiled = tf.tile(tf.reshape(losses, (-1, 1, 1, 1)), (1,) + initial_img.shape)
        grad_estimates.append(tf.reduce_mean(losses_tiled * noise, axis=0)/args.sigma)
        final_losses.append(losses)
    grad_estimate = tf.reduce_mean(grad_estimates, axis=0)
    final_losses = tf.concat(final_losses, axis=0)

    # GRADIENT ESTIMATION EVAL
    def get_grad(pt, spd, bs):
        num_batches = spd // bs
        losses = []
        grads = []
        feed_dict = {x: pt}
        for _ in range(num_batches):
            loss, dl_dx_ = sess.run([final_losses, grad_estimate], feed_dict)
            losses.append(np.mean(loss))
            grads.append(dl_dx_)
        return np.array(losses).mean(), np.mean(np.array(grads), axis=0)

    # CONCURRENT VISUALIZATION
    if args.visualize:
        with tf.device('/cpu:0'):
            render_feed = tf.placeholder(tf.float32, initial_img.shape)
            render_exp = tf.expand_dims(render_feed, axis=0)
            render_logits, _ = model(sess, render_exp)

    assert out_dir[-1] == '/'

    # HISTORY VARIABLES (for backtracking and momentum)
    num_queries = 0
    g = 0
    prev_adv = adv
    last_ls = []

    # STEP CONDITION (important for partial-info attacks)
    def robust_in_top_k(t_, prop_adv_,k_):
        if k == NUM_LABELS:
            return True
        for i in range(1):
            n = np.random.rand(*prop_adv_.shape)*args.sigma
            eval_logits_ = sess.run(eval_logits, {x: prop_adv_})[0]
            if not t_ in eval_logits_.argsort()[-k_:][::-1]:
                return False
        return True
        

    # MAIN LOOP
    for i in range(max_iters):
        start = time.time()
        if args.visualize:
            render_frame(sess, adv, i, render_logits, render_feed, out_dir)

        # CHECK IF WE SHOULD STOP
        padv = sess.run(eval_percent_adv, feed_dict={x: adv})
        if padv == 1 and epsilon <= goal_epsilon:
            print('[log] early stopping at iteration %d' % i)
            break

        prev_g = g
        l, g = get_grad(adv, args.samples_per_draw, batch_size)

        # SIMPLE MOMENTUM
        g = args.momentum * prev_g + (1.0 - args.momentum) * g

        # PLATEAU LR ANNEALING
        last_ls.append(l)
        last_ls = last_ls[-args.plateau_length:]
        if last_ls[-1] > last_ls[0] \
           and len(last_ls) == args.plateau_length:
            if max_lr > args.min_lr:
                print("[log] Annealing max_lr")
                max_lr = max(max_lr / args.plateau_drop, args.min_lr)
            last_ls = []

        # SEARCH FOR LR AND EPSILON DECAY
        current_lr = max_lr
        proposed_adv = adv - is_targeted * current_lr * np.sign(g)
        prop_de = 0.0
        if l < adv_thresh and epsilon > goal_epsilon:
            prop_de = delta_epsilon
        while current_lr >= args.min_lr:
            # PARTIAL INFORMATION ONLY
            if k < NUM_LABELS:
                proposed_epsilon = max(epsilon - prop_de, goal_epsilon)
                lower = np.clip(initial_img - proposed_epsilon, 0, 1)
                upper = np.clip(initial_img + proposed_epsilon, 0, 1)
            # GENERAL LINE SEARCH
            proposed_adv = adv - is_targeted * current_lr * np.sign(g)
            proposed_adv = np.clip(proposed_adv, lower, upper)
            num_queries += 1
            if robust_in_top_k(target_class, proposed_adv, k):
                if prop_de > 0:
                    delta_epsilon = max(prop_de, 0.1)
                    last_ls = []
                prev_adv = adv
                adv = proposed_adv
                epsilon = max(epsilon - prop_de/args.conservative, goal_epsilon)
                break
            elif current_lr >= args.min_lr*2:
                current_lr = current_lr / 2
                #print("[log] backtracking lr to %3f" % (current_lr,))
            else:
                prop_de = prop_de / 2
                if prop_de == 0:
                    raise ValueError("Did not converge.")
                if prop_de < 2e-3:
                    prop_de = 0
                current_lr = max_lr
                print("[log] backtracking eps to %3f" % (epsilon-prop_de,))

        # BOOK-KEEPING STUFF
        num_queries += args.samples_per_draw*(zero_iters if label_only else 1)
        log_text = 'Step %05d: loss %.4f lr %.2E eps %.3f (time %.4f)' % (i, l, \
                        current_lr, epsilon, time.time() - start)
        log_file.write(log_text + '\n')
        print(log_text)

        if i % log_iters == 0:
            lvq, lvs, lrvq, lrvs = sess.run([loss_vs_queries, loss_vs_steps,
                                             lr_vs_queries, lr_vs_steps], {
                                                 empirical_loss:l,
                                                 lr_placeholder:current_lr
                                             })
            writer.add_summary(lvq, num_queries)
            writer.add_summary(lrvq, num_queries)
            writer.add_summary(lvs, i)
            writer.add_summary(lrvs, i)

        if (i+1) % args.save_iters == 0 and args.save_iters > 0:
            np.save(os.path.join(out_dir, '%s.npy' % (i+1)), adv)
            scipy.misc.imsave(os.path.join(out_dir, '%s.png' % (i+1)), adv)
    log_output(sess, eval_logits, eval_preds, x, adv, initial_img, \
            target_class, out_dir, orig_class, num_queries)

if __name__ == '__main__':
    main()
