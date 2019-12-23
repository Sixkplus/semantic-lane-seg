import argparse
import tensorflow as tf
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import glob, os, time, sys

from tqdm import trange
from utils.config import Config
from model_swift import Swift_mobile, freetech_mobile_050, freetech_mobile_050_lane, freetech_mobile_025_lane


# //////////////////////////////////////////////////////////////
# /////////////////  SET UP CONFIGURATIONS /////////////////////
# //////////////////////////////////////////////////////////////

# mapping different model
model_config = {'freetech_mobile_050':freetech_mobile_050, 'freetech_mobile_050_lane':freetech_mobile_050_lane, 'freetech_mobile_025_lane_kd':freetech_mobile_025_lane}



dataset = 'freetech_lane'
filter_scale = 1

data_dir = 'ft_val'
    
class InferenceConfig(Config):
    def __init__(self, dataset, is_training, filter_scale):
        Config.__init__(self, dataset, is_training, filter_scale)
    
    # You can choose different model here
    model_type = 'Swift_mobile'
    model_type = 'freetech_mobile_050'
    model_type = 'freetech_mobile_050_lane'
    model_type = 'freetech_mobile_025_lane'
    model_type = 'freetech_mobile_025_lane_kd'
    ckpt_step = 600

    # Set pre-trained weights here (You can download weight from Google Drive) 
    #model_weight = './model/cityscapes/icnet_cityscapes_train_30k.npy'
    #model_weight = './useful_checkpoints/model.ckpt-28200'

    #model_weight = './snapshots/cityscapes/PSPNet18.ckpt-54200'

    #model_weight = './useful_checkpoints/' + model_type + '.ckpt-' + str(ckpt_step)
    model_weight = './snapshots/freetech/' + model_type + '_best.ckpt'
    #model_weight = './snapshots/freetech/' + model_type + '_alternate_best.ckpt'
    
    
    # Define default input size here
    INFER_SIZE = [1256, 1928, 3]
    reuse = False
                  
cfg = InferenceConfig(dataset, is_training=False, filter_scale=filter_scale)
cfg.display()

# -----------------------------------------------------------------------------------------------------------------

# //////////////////////////////////////////////////////////////
# //////// Create graph, session, and restore weights //////////
# //////////////////////////////////////////////////////////////


# Create graph here 
model = model_config[cfg.model_type]
net = model(cfg=cfg, mode='inference')

# Create session & restore weight!
net.create_session()
net.restore(cfg.model_weight)

run_meta = tf.RunMetadata()
opts = tf.profiler.ProfileOptionBuilder.float_operation()
flops = tf.profiler.profile(tf.get_default_graph(), run_meta=run_meta, cmd='op', options=opts)
print(flops.total_float_ops)

# -----------------------------------------------------------------------------------------------------------------

# //////////////////////////////////////////////////////////////
# ///////////////// Run Inference on datasets //////////////////
# //////////////////////////////////////////////////////////////

    
test_images = glob.glob(os.path.join(data_dir, '*.png'))


ind = 0
for imgName in test_images:
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_images)))
    sys.stdout.flush()
    #tot_st = time.time()
    
    img = cv2.imread(imgName)

    rand_h = np.random.randint(1256 - cfg.INFER_SIZE[0] + 1)
    rand_w = np.random.randint(1928 - cfg.INFER_SIZE[1] + 1)

    img = img[rand_h:rand_h+cfg.INFER_SIZE[0], rand_w:rand_w+cfg.INFER_SIZE[1], :]
    
    st = time.time()
    color_and_id_seg_lane = net.predict_color_and_id_seg_lane(img)
    #result_id = net.predict_id(img)[0]
    proc_time = time.time() - st
    print("  Processing time: ", proc_time)

    result_color, result_id = color_and_id_seg_lane[0][0], color_and_id_seg_lane[1][0]
    lane_color, lane_id = color_and_id_seg_lane[2][0], color_and_id_seg_lane[3][0]
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result_alpha = 0.4 * img + 0.3 * result_color + 0.4 * lane_color
    #cv2.imwrite(os.path.join('./Test', imgName.split('/')[-1].replace('.png', '_color.png')), cv2.cvtColor(np.uint8(result_color), cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join('./Test_ab', imgName.split('/')[-1].replace('.png', '_ab.png')), cv2.cvtColor(np.uint8(result_alpha), cv2.COLOR_RGB2BGR))
    #cv2.imwrite(os.path.join('./Test_id', imgName.split('/')[-1].replace('.png', '_id.png')), result_id)
    
    #tot_proc_time = time.time() - tot_st
    #print("  Total time: ", tot_proc_time)
    
    ind += 1
    