import scipy.io as sio
import numpy as np
import tensorflow as tf

label_colours_cityscapes = [[128, 64, 128], [244, 35, 231], [69, 69, 69]
                # 0 = road, 1 = sidewalk, 2 = building
                ,[102, 102, 156], [190, 153, 153], [153, 153, 153]
                # 3 = wall, 4 = fence, 5 = pole
                ,[250, 170, 29], [219, 219, 0], [106, 142, 35]
                # 6 = traffic light, 7 = traffic sign, 8 = vegetation
                ,[152, 250, 152], [69, 129, 180], [219, 19, 60]
                # 9 = terrain, 10 = sky, 11 = person
                ,[255, 0, 0], [0, 0, 142], [0, 0, 69]
                # 12 = rider, 13 = car, 14 = truck
                ,[0, 60, 100], [0, 79, 100], [0, 0, 230]
                # 15 = bus, 16 = train, 17 = motocycle
                ,[119, 10, 32]]
                # 18 = bicycle

label_colours_camvid = [
    # 0 = building, 1 = tree, 2 = sky
    [128, 0, 0], [128,128,0], [128,128,128], 
    # 3 = car, 4 = signsymbol, 5 = road
    [64,0,128], [192,128,128], [128,64,128], 
    # 6 = pedestrian, 7 = fence, 8 = column_pole
    [64,64,0], [64,64,128], [192,192,128], 
    # 9 = sidewalk, 10 = bicyclist
    [0,0,192], [0,128,192]
]

label_colours_freetech = [[153,153,153], [128, 64, 128], [0, 0, 142]
                # 0 = background, 1 = road, 2 = vehicle
                ,[255, 0, 0], [219, 19, 60], [219, 219, 0]]
                # 3 = rider, 4 = walker, 5 = cone

label_colours_freetech_lane = [ [0,0,0], [255,255,0] ]

label_colours_agric = [[153,153,153], [255,0,0], [219, 219, 0]
                # 0 = background, 1 = 烤烟, 2 = 玉米
                ,[0, 0, 255]]
                # 3 = 薏米仁

id_list_cityscapes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
id_list_freetech = [0,1,2,3,4,5]
id_list_freetech_lane = [0,1]
id_list_agric = [0,1,2,3]
id_list_camvid = [0,1,2,3,4,5,6,7,8,9,10]


matfn = './utils/color150.mat'
def read_labelcolours(matfn):
    mat = sio.loadmat(matfn)
    color_table = mat['colors']
    shape = color_table.shape
    color_list = [tuple(color_table[i]) for i in range(shape[0])]

    return color_list

def decode_labels(mask, img_shape, num_classes):
    if num_classes == 150:
        color_table = read_labelcolours(matfn)
    elif num_classes == 6:
        color_table = label_colours_freetech
    elif num_classes == 2:
        color_table = label_colours_freetech_lane
    elif num_classes == 4:
        color_table = label_colours_agric
    elif num_classes == 11:
        color_table = label_colours_camvid
    else:
        color_table = label_colours_cityscapes

    color_mat = tf.constant(color_table, dtype=tf.float32)
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, int(img_shape[0]), int(img_shape[1]), 3))
    #pred = tf.reshape(pred, (1, 1024, 2048, 3))
    #pred = tf.reshape(pred, (1, 256, 256, 3))
    
    return pred

def decode_ids(mask, img_shape, num_classes):

    if num_classes == 6:
        id_list = id_list_freetech
    elif num_classes == 2:
        id_list = id_list_freetech_lane
    elif num_classes ==4:
        id_list = id_list_agric
    elif num_classes == 11:
        id_list = id_list_camvid
    else:
        id_list = id_list_cityscapes
    id_mat = tf.constant(id_list, dtype=tf.float32, shape = (num_classes, 1))
    onehot_output = tf.one_hot(mask, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, id_mat)
    pred = tf.reshape(pred, (1, int(img_shape[0]), int(img_shape[1]), 1))
    #pred = tf.reshape(pred, (1, 1024, 2048, 1))
    #pred = tf.reshape(pred, (1, 256, 256, 1))
    
    return pred


def prepare_label(input_batch, new_size, num_classes, one_hot=True):
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size) # as labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reducing the channel dimension.
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
            
    return input_batch
