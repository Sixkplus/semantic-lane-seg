import argparse
import tensorflow as tf
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import glob, os, time, sys
from skimage.io import imread

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
    
class InferenceConfig(Config):
    def __init__(self, dataset, is_training, filter_scale):
        Config.__init__(self, dataset, is_training, filter_scale)
    
    # You can choose different model here
    model_type = 'Swift_mobile'
    model_type = 'freetech_mobile_050_lane'
    #model_type = 'freetech_mobile_050_lane'
    #model_type = 'freetech_mobile_025_lane'
    #model_type = 'freetech_mobile_025_lane_kd'
    ckpt_step = 600

    #model_weight = './useful_checkpoints/model.ckpt-28200'

    #model_weight = './useful_checkpoints/' + model_type + '.ckpt-' + str(ckpt_step)
    model_weight = './snapshots/freetech/' + model_type + '_best.ckpt'
    #model_weight = './snapshots/freetech/' + model_type + '_alternate_best.ckpt'

    # Define default input size here
    INFER_SIZE = [1256, 1928, 3]
    FEAT_SIZE = [314, 482, 19]
    reuse = False

                  
cfg = InferenceConfig(dataset, is_training=False, filter_scale=filter_scale)
cfg.display()

# -----------------------------------------------------------------------------------------------------------------

# //////////////////////////////////////////////////////////////
# //////// Create graph, session, and restore weights //////////
# //////////////////////////////////////////////////////////////


# Create graph here 
model = model_config[cfg.model_type]

# Create session & restore weight!
net = model(cfg=cfg, mode='inference')
net.create_session()
net.restore(cfg.model_weight)

# -----------------------------------------------------------------------------------------------------------------

# //////////////////////////////////////////////////////////////
# ///////////////// Run Inference on datasets //////////////////
# //////////////////////////////////////////////////////////////


def get_xyindex(h,w):
    index_list = []
    for i in range(h):
        for j in range(w):
            index_list.append([j,i])
    return np.array(index_list)

def get_batchindex(b,h,w):
    index_list = []
    for k in range(b):
        for i in range(h):
            for j in range(w):
                index_list.append([k])
    return np.array(index_list)

def warp(key_feature, flow):
    shape = flow.get_shape().as_list()
    #key_feature = tf.image.resize_bilinear(key_feature, shape[1:3])
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]
    with tf.name_scope('warp') as scope:
        #flow_index = flow + tf.constant(get_xyindex(height, width),shape=[height, width, 2],dtype=tf.float32)
        flow_index = flow + xyindex
        # This should be a number between 0 and height/width
        flow_index = tf.minimum(flow_index, [width-1,height-1])
        flow_index = tf.maximum(flow_index, [0.0,0.0])
        #batch_index = tf.constant(get_batchindex(batch_size, height, width),shape=[batch_size, height, width, 1],dtype=tf.float32)
        x_index = tf.reshape(flow_index[:,:,:,0], [batch_size, height, width, 1])
        y_index = tf.reshape(flow_index[:,:,:,1], [batch_size, height, width, 1])
        x_floor = tf.floor(x_index)
        x_ceil = tf.ceil(x_index)
        y_floor = tf.floor(y_index)
        y_ceil = tf.ceil(y_index)
        flow_index_ff = tf.cast(tf.concat([batch_index,y_floor,x_floor], 3), tf.int32)
        flow_index_cf = tf.cast(tf.concat([batch_index,y_ceil,x_floor], 3), tf.int32)
        flow_index_fc = tf.cast(tf.concat([batch_index,y_floor,x_ceil], 3), tf.int32)
        flow_index_cc = tf.cast(tf.concat([batch_index,y_ceil,x_ceil], 3), tf.int32)

        # bi-linear interpolation
        thetax = x_index - x_floor
        _thetax = 1.0 - thetax
        thetay =  y_index - y_floor
        _thetay = 1.0 - thetay
        
        coeff_ff = _thetax * _thetay
        coeff_cf = _thetax * thetay
        coeff_fc = thetax * _thetay
        coeff_cc = thetax * thetay
        ff = tf.gather_nd(key_feature, flow_index_ff) * coeff_ff
        cf = tf.gather_nd(key_feature, flow_index_cf) * coeff_cf
        fc = tf.gather_nd(key_feature, flow_index_fc) * coeff_fc
        cc = tf.gather_nd(key_feature, flow_index_cc) * coeff_cc
        warp_image = tf.add_n([ff,cf,fc,cc])
    return warp_image
    


batch_index = tf.constant(get_batchindex(1, cfg.FEAT_SIZE[0], cfg.FEAT_SIZE[1]),shape=[1, cfg.FEAT_SIZE[0], cfg.FEAT_SIZE[1], 1],dtype=tf.float32)
xyindex = tf.constant(get_xyindex(cfg.FEAT_SIZE[0], cfg.FEAT_SIZE[1]),shape=[1, cfg.FEAT_SIZE[0], cfg.FEAT_SIZE[1], 2],dtype=tf.float32)

feature_placeholder = tf.placeholder(dtype = tf.float32, shape = [1, cfg.FEAT_SIZE[0], cfg.FEAT_SIZE[1], cfg.param['num_classes']])
feature_lane_placeholder = tf.placeholder(dtype = tf.float32, shape = [1, cfg.FEAT_SIZE[0], cfg.FEAT_SIZE[1], 2])
flow_placeholder = tf.placeholder(dtype = tf.float32, shape = [1, cfg.FEAT_SIZE[0], cfg.FEAT_SIZE[1], 2])

warp_feature = warp(feature_placeholder, flow_placeholder)
warp_feature_lane = warp(feature_lane_placeholder, flow_placeholder)


GOP_FRAMES_NUM = 12
np.random.seed(5)

tot = []

val_folders = ['video_test']
for VAL_FOLDER in val_folders:
    PATH_MV = os.path.join(VAL_FOLDER, 'mv_cont')
    PATH_RES = os.path.join(VAL_FOLDER, 'res_cont')
    PATH_IMG = os.path.join(VAL_FOLDER, 'img_cont')

    img_names = glob.glob(PATH_IMG + '/*.png')
    n = len(img_names)

    flow = np.zeros([1, cfg.FEAT_SIZE[0], cfg.FEAT_SIZE[1], 2])
    feature = np.zeros([1, cfg.FEAT_SIZE[0], cfg.FEAT_SIZE[1], cfg.param['num_classes']])
    feature_lane = np.zeros([1, cfg.FEAT_SIZE[0], cfg.FEAT_SIZE[1], 2])
    cum_res = 0

    for ind in range(n):
        img = cv2.imread(PATH_IMG + '/frame' + str(ind) + '.png')
        # I-frame
        if ind % 12 == 0:
            st = time.time()
            feature, feature_lane = net.predict_feature(img)
            print( 'Frame id: ',ind, ", I-frame, processing time: ", time.time()-st)
            cum_res = 0
        
        # P-frame
        else:
            res = imread(PATH_RES + '/frame'+ str(ind) + '.png').astype(np.float32)
            res = abs(res * 2 - 256)
            res = np.sum(res, axis=2, keepdims=True)
            cum_res += res.sum()
            
            if(res.sum() > 36000000 or cum_res > 90000000):
                st1 = time.time()
                feature, feature_lane = net.predict_feature(img)
                cum_res = 0
                proc_time = time.time()-st1
                print('Frame id: ',ind, ", P-frame, processing time: ", proc_time)
                tot.append(1)
            else:
                save_mvPng = imread(PATH_MV + '/frame'+ str(ind) + '.png' ).astype(np.int16)
                flow_origin = np.array([ (save_mvPng[:,:,0] << 8) + (save_mvPng[:,:,1]), (save_mvPng[:,:,2] << 8) + (save_mvPng[:,:,3]) ])
                flow_origin = np.transpose(flow_origin, [1,2,0]).reshape(1, cfg.INFER_SIZE[0], cfg.INFER_SIZE[1], 2)
                flow_origin -= 2048

                flow[0,:,:,0] = cv2.resize(np.float32(flow_origin[0,:,:,0]), (0,0), fx=cfg.FEAT_SIZE[1]/cfg.INFER_SIZE[1], fy=cfg.FEAT_SIZE[0]/cfg.INFER_SIZE[0], interpolation = cv2.INTER_LINEAR)*cfg.FEAT_SIZE[0]/cfg.INFER_SIZE[0]
                flow[0,:,:,1] = cv2.resize(np.float32(flow_origin[0,:,:,1]), (0,0), fx=cfg.FEAT_SIZE[1]/cfg.INFER_SIZE[1], fy=cfg.FEAT_SIZE[0]/cfg.INFER_SIZE[0], interpolation = cv2.INTER_LINEAR)*cfg.FEAT_SIZE[1]/cfg.INFER_SIZE[1]
                
                flow_origin = abs(flow_origin.reshape(cfg.INFER_SIZE[0], cfg.INFER_SIZE[1], 2))
                flow_origin = np.sum(flow_origin, axis=2, keepdims=True)
                st1 = time.time()
                feature = net.sess.run(warp_feature, feed_dict={feature_placeholder: feature, flow_placeholder: -flow})
                feature_lane = net.sess.run(warp_feature_lane, feed_dict={feature_lane_placeholder: feature_lane, flow_placeholder: -flow})
                proc_time = time.time()-st1
                print('Frame id: ',ind, ", P-frame, processing time: ", proc_time)
                tot.append(0)
        
        
        color_and_id = net.predict_color_from_feature(feature, feature_lane)
        result_color, result_id = color_and_id[0][0], color_and_id[1][0]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result_alpha = 0.5 * img + 0.5 * result_color

        if not os.path.exists('video_output'):
            os.makedirs('video_output')
        
        cv2.imwrite('video_output/frame' + str(ind) +'.png', cv2.cvtColor(np.uint8(result_alpha), cv2.COLOR_RGB2BGR))
        #cv2.imwrite('output_id/' + cur_dir_name.split("/")[-1]+'.png', result_id)
        
tot = np.array(tot)

print(len(tot), sum(tot))
    