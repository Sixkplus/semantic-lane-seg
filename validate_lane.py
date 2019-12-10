import argparse
import time, cv2, os, shutil, glob

import tensorflow as tf
import numpy as np
from tqdm import trange

from utils.config import Config
from utils.image_lane_reader import ImageReader, read_labeled_image_list
from model_swift import Swift_mobile, freetech_mobile_050, freetech_mobile_050_lane
from utils.visualize import decode_labels, decode_ids

# mapping different model
model_config = {'Swift_mobile':Swift_mobile, 'freetech_mobile_050':freetech_mobile_050, 'freetech_mobile_050_lane':freetech_mobile_050_lane}

def get_arguments():
    parser = argparse.ArgumentParser(description="Semantic Segmentation models Eval")

    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes', 'freetech_day', 'freetech_lane', 'agric'],
                        required=True)
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])

    return parser.parse_args()

class EvalConfig(Config):
    def __init__(self, dataset, filter_scale=1):
        Config.__init__(self, dataset, filter_scale=filter_scale)

    # Set pre-trained weights here (You can download weight using `python script/download_weights.py`) 
    #model_type = 'ICNet'
    #model_type = 'PSPNet18'
    #model_type = 'Swift18'
    #model_type = 'Swift18_light_BN'
    #model_type = 'Swift18_light'
    model_type = 'Swift_mobile'
    model_type = 'freetech_mobile_050_lane'

    ckpt_step = 100

    model_weight = './snapshots/freetech_lane/' + model_type + '.ckpt-' + str(ckpt_step)

    #model_weight = './snapshots_restart/cityscapes/' + model_type + '.ckpt-' + str(ckpt_step)

    # Set hyperparameters here, you can get much more setting in Config Class, see 'utils/config.py' for details.
    BATCH_SIZE = 1
    N_WORKERS = 2
    reuse = False
    LABEL_SHAPE = [1256, 1928]


def eval(cfg):
    img_names, _, _ = read_labeled_image_list(cfg.FREETECH_DATA_DIR, cfg.freetech_lane_eval_list)
    
    model = model_config[cfg.model_type]
    reader = ImageReader(cfg=cfg, mode='eval')
    net = model(image_reader=reader, cfg=cfg, mode='eval')
    
    # mIoU
    pred_flatten = tf.reshape(net.output, [-1,])
    label_flatten = tf.reshape(net.labels, [-1,])
    
    output_id = decode_ids(pred_flatten, cfg.LABEL_SHAPE, cfg.param['num_classes'])
    output_color = decode_labels(pred_flatten, cfg.LABEL_SHAPE, cfg.param['num_classes'])

    mask = tf.not_equal(label_flatten, cfg.param['ignore_label'])
    indices = tf.squeeze(tf.where(mask), 1)
    gt = tf.cast(tf.gather(label_flatten, indices), tf.int32)
    pred = tf.gather(pred_flatten, indices)

    if cfg.dataset == 'ade20k':
        pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes']+1)
    elif cfg.dataset == 'cityscapes':
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])
    else:
        mIoU, update_op = tf.metrics.mean_iou(predictions=pred, labels=gt, num_classes=cfg.param['num_classes'])
    
    net.create_session()
    net.restore(cfg.model_weight)
    
    for i in trange(cfg.param['eval_steps'], desc='evaluation', leave=True):
        _ = net.sess.run([update_op])
           
    #print('mIoU: {}'.format(net.sess.run(mIoU)))
    cur_MIOU = net.sess.run(mIoU)
    net.sess.close()
    tf.reset_default_graph()
    return cur_MIOU


def main():
    args = get_arguments()  
    cfg = EvalConfig(dataset=args.dataset, filter_scale=args.filter_scale)

    cfg.param['ignore_label'] = 255
    
    ckpt_step = cfg.ckpt_step

    max_mIoU = 0.00

    while(True):
        while(ckpt_step < 160000):
            cfg.model_weight = './snapshots/freetech_lane/' + cfg.model_type + '.ckpt-' + str(ckpt_step) 
            if os.path.exists(cfg.model_weight + '.meta'):
                mIoU = eval(cfg)
                if(mIoU > max_mIoU):
                    max_mIoU = mIoU
                    # data, meta, index
                    to_save_files = glob.glob(cfg.model_weight+'*')
                    for file in to_save_files:
                        new_file = file.replace(cfg.model_type + '.ckpt-' + str(ckpt_step), \
                            cfg.model_type + '_best' + '.ckpt')
                        shutil.copy(file, new_file)
                print(max_mIoU, mIoU)
                ckpt_step += 100
            else:
                time.sleep(5)

if __name__ == '__main__':
    main()
