"""
This code is based on DrSleep's framework: https://github.com/DrSleep/tensorflow-deeplab-resnet 
"""
import argparse
import os
import sys
import time

import tensorflow as tf
import numpy as np
from tensorflow.python.client import device_lib
from tqdm import trange

from utils.config import Config
from utils.visualize import decode_labels
from utils.image_reader import ImageReader, prepare_label

from model_swift import Swift_mobile, freetech_mobile_050

local_device_protos = device_lib.list_local_devices()

def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced ICNet")
    
    parser.add_argument("--random-mirror", required=False, action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--use-weight", required=False, action="store_true",
                        help="Whether to add class bias weights during the training.")
    parser.add_argument("--random-scale", required=False, action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--update-mean-var", action="store_true",
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true",
                        help="whether to train beta & gamma in bn layer")
    parser.add_argument("--dataset", required=True,
                        help="Which dataset to trained with",
                        choices=['cityscapes', 'ade20k', 'freetech', 'freetech_night', 'freetech_special', 'agric', 'others'])
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    return parser.parse_args()

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, axis = 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def get_mask(gt, num_classes, ignore_label):
    less_equal_class = tf.less_equal(gt, num_classes-1)
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)

    return indices

def create_loss(output, label, num_classes, ignore_label, USE_WEIGHT=False, UP_PRED=True, BOOT_STRAP=True):
    if UP_PRED:
        output = tf.image.resize_bilinear(output, size=tf.stack(label.get_shape()[1:3]))
    raw_pred = tf.reshape(output, [-1, num_classes])
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1,])

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)

    if USE_WEIGHT:
        #----------------------Weight---------------------------------#
        gt = tf.one_hot(gt, num_classes, 1., 0., dtype = tf.float32)
        
        probs = np.array([0.36880976, 0.06086874, 0.2281957 , 0.0065593 , 0.00878296,
                        0.01227566, 0.00208491, 0.00552862, 0.15917384, 0.01158665,
                        0.04011515, 0.01217303, 0.00134857, 0.07001108, 0.00267634,
                        0.00235406, 0.00233017, 0.00098647, 0.00413899], dtype = np.float32)
        c = 1.2
        weights = 1/np.log(probs+c)
        weights = weights.reshape([1,num_classes])
        #weights = weights/weights.sum() * num_classes
        
        weights = tf.constant(weights, dtype = tf.float32)
        weights = tf.reduce_sum(weights * gt, axis=-1)
        with tf.device("/cpu:0"):
            loss = tf.losses.softmax_cross_entropy(onehot_labels=gt, logits=pred, weights=weights)
        #----------------------Weight---------------------------------#
    
    else:
        # No Weight
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)

    min_loss_num = tf.math.floordiv(tf.shape(loss)[0], 16)

    #loss_sorted, _ = tf.sort(loss, axis=-1, direction='DESCENDING')

    if BOOT_STRAP:
        loss_top_k, loss_top_k_indices = tf.math.top_k(loss, min_loss_num)
        greater_equal_mask = tf.greater_equal(loss, -np.log(0.7))
        greater_equal_mask = tf.squeeze(tf.where(greater_equal_mask), 1)
        # Take  max(min_loss_num, )
        loss = tf.cond(tf.greater_equal(loss_top_k[-1], -np.log(0.7)), lambda:tf.gather(loss, greater_equal_mask), lambda:tf.gather(loss, loss_top_k_indices))
        #loss = tf.gather(loss, greater_equal_mask)

    reduced_loss = tf.reduce_mean(loss)

    return reduced_loss


def create_losses(net, label, cfg):
    if cfg.AUX_LOSSES == False:
        # Get output from the output layer
        pred = net.layers['conv6_cls']
        # soft-max cross-entropy
        loss_cls = create_loss(pred, label, cfg.param['num_classes'], cfg.param['ignore_label'], cfg.USE_WEIGHT)
    else:
        # Get output from different branches
        pred_1 = net.layers['conv6_cls']
        pred_2 = net.layers['conv5_cls']
        pred_3 = net.layers['conv4_cls']
        # soft-max cross-entropy
        loss_cls_1 = create_loss(pred_1, label, cfg.param['num_classes'], cfg.param['ignore_label'], cfg.USE_WEIGHT)
        loss_cls_2 = create_loss(pred_2, label, cfg.param['num_classes'], cfg.param['ignore_label'], cfg.USE_WEIGHT)
        loss_cls_3 = create_loss(pred_3, label, cfg.param['num_classes'], cfg.param['ignore_label'], cfg.USE_WEIGHT)
        loss_cls = 1.0 * loss_cls_1 + 0.4 * loss_cls_2 + 0.2 * loss_cls_3
    # l2-regularization
    l2_losses = [cfg.WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    
    reduced_loss = loss_cls + tf.add_n(l2_losses)
    return loss_cls, reduced_loss

Models = {'Swift_mobile':Swift_mobile, 'freetech_mobile_050':freetech_mobile_050}

class TrainConfig(Config):
    def __init__(self, dataset, is_training,  filter_scale=1, use_weight=False, random_scale=None, random_mirror=None):
        Config.__init__(self, dataset, is_training, filter_scale, random_scale, random_mirror)
        self.USE_WEIGHT = use_weight

    #model = 'PSPNet18'
    #model = 'Swift18'
    #model = 'Swift18_light'
    #model = 'Swift50_light_BN'
    model = 'Swift_mobile'
    model = 'freetech_mobile_050'
    
    
    # Continue Training
    #model_weight = './snapshots/cityscapes/Swift18_light_BN_best.ckpt'
    #model_weight = './useful_checkpoints/Swift50_light_BN_best.ckpt'
    model_weight = './snapshots/cityscapes/Swift_mobile.ckpt-5700'

    RESTORE = False

    #pretrained_weight = './pretrained/resnet_v1_50.ckpt'
    #pretrained_weight = './pretrained/mobilenet_v1_1.0_224.ckpt'
    pretrained_weight = './pretrained/mobilenet_v1_0.5_160.ckpt'
    PRETRAINED = True

    # Directory to save the checkpoints
    SNAPSHOT_DIR = './snapshots'
    
    # Set hyperparameters here, you can get much more setting in Config Class, see 'utils/config.py' for details.
    TRAINING_SIZE = [1256, 1256] 
    BATCH_SIZE = 8
    N_WORKERS = 2
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-7
    MOMENTUM = 0.9
    POWER = 0.9
    TRAINING_STEPS = 60000 + 1
    SAVE_PRED_EVERY = 100
    reuse = False

    # The ratio:  gradient change of FINE TUNE part : learning_rate * fine_tune_ratio
    lr_lager_ratio = 1.0
    ft_ratio = 1e-2

    # Multi-layer auxiliary loss
    AUX_LOSSES = False

    # Parameters for multi-gpu training
    gpu_list = [x.name for x in local_device_protos if x.device_type == 'GPU']
    num_gpu = len(gpu_list)
    BATCH_SIZE = BATCH_SIZE * num_gpu

def main():
    """Create the model and start the training."""
    args = get_arguments()

    """
    Get configurations here. We pass some arguments from command line to init configurations, for training hyperparameters, 
    you can set them in TrainConfig Class.

    Note: we set filter scale to 1 for pruned model, 2 for non-pruned model. The filters numbers of non-pruned
          model is two times larger than prunde model, e.g., [h, w, 64] <-> [h, w, 32].
    """
    cfg = TrainConfig(dataset=args.dataset, 
                is_training=True,
                use_weight=args.use_weight,
                random_scale=args.random_scale,
                random_mirror=args.random_mirror,
                filter_scale=args.filter_scale)
    cfg.display()

    # Using Poly learning rate policy 
    base_lr = tf.constant(cfg.LEARNING_RATE)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / cfg.TRAINING_STEPS), cfg.POWER))

    tf.summary.scalar('LearningRate', learning_rate)

    # Setup training network and training samples
    train_reader = ImageReader(cfg=cfg, mode='train')
    
    my_optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    #my_optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum=cfg.MOMENTUM) 

    next_img, next_label = train_reader.next_image, train_reader.next_label
    image_splits = tf.split(next_img, cfg.num_gpu)
    label_splits = tf.split(next_label, cfg.num_gpu)
    tower_grads = []
    tower_loss = []
    tower_loss_cls = []
    counter = 0
    with tf.variable_scope(tf.get_variable_scope()):
        for d in range(cfg.num_gpu):
            with tf.device('/gpu:%d' % d):
                with tf.name_scope('tower_%d' % d):
                    model = Models[cfg.model]
                    train_net = model(next_image = image_splits[counter], next_label = label_splits[counter], cfg=cfg, mode='train')
                    loss_cls, reduced_loss = create_losses(train_net, train_net.labels, cfg)
                    counter += 1
                    # Create loss
                    tf.get_variable_scope().reuse_variables()
                    all_trainable = [v for v in tf.trainable_variables() if (('beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma) and 'Logits' not in v.name]
                    with tf.variable_scope("loss"):
                        grads = my_optimizer.compute_gradients(reduced_loss, all_trainable)
                        tower_grads.append(grads)
                        tower_loss.append(reduced_loss)
                        tower_loss_cls.append(loss_cls)
    
    
    for i in range(cfg.num_gpu):
        tf.summary.scalar("Losses/total_loss", tower_loss[i])
        tf.summary.scalar("Losses/cls_loss", tower_loss_cls[i])

    mean_loss = tf.stack(axis=0, values=tower_loss)
    mean_loss = tf.reduce_mean(mean_loss, 0)

    # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
    if args.update_mean_var == False:
        update_ops = None
    else:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    with tf.control_dependencies(update_ops):
        mean_grads = average_gradients(tower_grads)
        pretrained_var = [v for v in tf.trainable_variables() if 'Mobilenet' in v.name]
        lr_lager_vars = [v for v in tf.trainable_variables() if 'conv6_cls' in v.name]
        # The fine-tune part learning rate is smaller
        if(cfg.ft_ratio != 1):
            for idx, grad_and_var in enumerate(mean_grads):
                grad, var = grad_and_var[0], grad_and_var[1]
                if var in pretrained_var:
                    mean_grads[idx] = (mean_grads[idx][0] * cfg.ft_ratio, var)

        if(cfg.lr_lager_ratio != 1):
            for idx, grad_and_var in enumerate(mean_grads):
                grad, var = grad_and_var[0], grad_and_var[1]
                if var in lr_lager_vars:
                    mean_grads[idx] = (mean_grads[idx][0] * cfg.lr_lager_ratio, var)

        train_op = my_optimizer.apply_gradients(mean_grads)
        #train_op = slim.learning.create_train_op()

    # Setup validation network and validation samples
    '''
    with tf.variable_scope('', reuse=True):
        val_reader = ImageReader(cfg, mode='eval')
        val_net = ICNet(image_reader=val_reader, 
                            cfg=cfg, mode='train')

        val_loss_cls, val_reduced_loss = create_losses(val_net, val_net.labels, cfg)
    '''

    # Set restore variable 
    #restore_var = [v for v in tf.trainable_variables() if ('Momentum' not in v.name and 'conv6_cls' not in v.name]
    restore_var = [v for v in tf.trainable_variables() if 'conv6_cls' not in v.name]
    
    
    # Create session & restore weights (Here we only need to use train_net to create session since we reuse it)
    train_net.create_session()

    # tensorboard
    merge_summary = tf.summary.merge_all()

    if not os.path.exists('logs/'+cfg.model):
        os.mkdir('logs/'+cfg.model)
    train_writer = tf.summary.FileWriter('logs/'+cfg.model)
    

    # Load the pretrained model
    if cfg.PRETRAINED:
        pretrained_var = [v for v in tf.trainable_variables() if 'Mobilenet' in v.name and 'Logits' not in v.name]
        train_net.restore(cfg.pretrained_weight, pretrained_var)
        
    # Training from the former checkpoint
    if cfg.RESTORE:
        #train_net.restore(cfg.model_weight, restore_var)
        train_net.restore(cfg.model_weight)
    
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

    # Iterate over training steps.
    for step in trange(cfg.TRAINING_STEPS):
        start_time = time.time()
            
        feed_dict = {step_ph: step}
        train_summary, loss_value, loss3, _ = train_net.sess.run([merge_summary, mean_loss, tower_loss_cls, train_op], feed_dict=feed_dict)
        duration = time.time() - start_time
        loss3 = np.average(loss3)
        print('step {:d} \t total loss = {:.5f}, cls_loss = {:.5f} ({:.3f} sec/step)'.\
                format(step, loss_value, loss3, duration))

        if(step % 20 == 0):
            train_writer.add_summary(train_summary,step)
        
        # Save the model
        if step % cfg.SAVE_PRED_EVERY == 0 and step != 0:
            train_net.save(saver, cfg.SNAPSHOT_DIR, step)

        
if __name__ == '__main__':
    main()
