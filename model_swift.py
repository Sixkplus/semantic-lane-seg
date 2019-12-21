import tensorflow as tf
from network import Network
from utils.image_reader import _infer_preprocess
from utils.visualize import decode_labels, decode_ids
import numpy as np

from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v1

import mobilenet_v1



class Swift_mobile(Network):
    def __init__(self, cfg, mode, image_reader=None, next_image = None, next_label = None):
        self.cfg = cfg
        self.mode = mode
        '''
        if mode == 'train':
            self.images, self.labels = image_reader.next_image, image_reader.next_label    
        
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)
        '''
        if mode == 'train':
            self.images, self.labels = next_image, next_label    
        
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)

        elif mode == 'eval':
            self.images, self.labels = image_reader.next_image, image_reader.next_label    
        
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)
            
            self.output = self.get_output_node()

        elif mode == 'inference':
            # Create placeholder and pre-process here.
            self.img_placeholder = tf.placeholder(dtype=tf.float32, shape=cfg.INFER_SIZE)
            self.logits_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.cfg.param['num_classes']])
            self.images, self.o_shape, self.n_shape = _infer_preprocess(self.img_placeholder, cfg.IMG_MEAN)
            
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)

            self.output = self.get_output_node()
            self.output_id = self.get_output_id()

            self.output_feature = self.layers['conv6_cls']

            self.output_from_feature = self.get_output_node_from_feature()
            self.output_id_from_feature = self.get_output_id_from_feature()

    def get_output_node(self):
        if self.mode == 'inference':
            # Get logits from final layer
            logits = self.layers['conv6_cls']

            # Upscale the logits and decode prediction to get final result.
            logits_up = tf.image.resize_bilinear(logits, size=self.o_shape, align_corners=True)
            #logits_up = tf.image.crop_to_bounding_box(logits_up, 0, 0, self.o_shape[0], self.o_shape[1])

            output_classes = tf.argmax(logits_up, axis=3)
            output = decode_labels(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])

        elif self.mode == 'eval':
            logits = self.layers['conv6_cls']

            logits_up = tf.image.resize_bilinear(logits, size=tf.shape(self.labels)[1:3], align_corners=True)
            output = tf.argmax(logits_up, axis=3)
            output = tf.expand_dims(output, axis=3)

        return output
    
    def get_output_id(self):
        if self.mode == 'inference':
            # Get logits from final layer
            logits = self.layers['conv6_cls']

            # Upscale the logits and decode prediction to get final result.
            logits_up = tf.image.resize_bilinear(logits, size=self.o_shape, align_corners=True)
            #logits_up = tf.image.crop_to_bounding_box(logits_up, 0, 0, self.o_shape[0], self.o_shape[1])

            output_classes = tf.argmax(logits_up, axis=3)
            output = decode_ids(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])

        elif self.mode == 'eval':
            logits = self.layers['conv6_cls']

            logits_up = tf.image.resize_bilinear(logits, size=tf.shape(self.labels)[1:3], align_corners=True)
            output = tf.argmax(logits_up, axis=3)
            output = tf.expand_dims(output, axis=3)

        return output

    def predict_feature(self, image):
        # The output layer
        return self.sess.run(self.output_feature, feed_dict = {self.img_placeholder: image})

    def predict(self, image):
        return self.sess.run(self.output, feed_dict={self.img_placeholder: image})
    
    def predict_id(self, image):
        return self.sess.run(self.output_id, feed_dict={self.img_placeholder: image})
    
    def predict_color_and_id(self, image):
        #output_cmb = self.sess.run([self.output, self.output_id], feed_dict={self.img_placeholder: image})
        return self.sess.run([self.output, self.output_id], feed_dict={self.img_placeholder: image})
    
    def get_output_id_from_feature(self):
        # Upscale the logits and decode prediction to get final result.
        logits_up = tf.image.resize_bilinear(self.logits_placeholder, size=self.o_shape, align_corners=True)
        #logits_up = tf.image.crop_to_bounding_box(logits_up, 0, 0, self.o_shape[0], self.o_shape[1])

        output_classes = tf.argmax(logits_up, axis=3)
        output_id = decode_ids(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])
        return output_id
    
    def get_output_node_from_feature(self):
        # Upscale the logits and decode prediction to get final result.
        logits_up = tf.image.resize_bilinear(self.logits_placeholder, size=self.o_shape, align_corners=True)
        #logits_up = tf.image.crop_to_bounding_box(logits_up, 0, 0, self.o_shape[0], self.o_shape[1])

        output_classes = tf.argmax(logits_up, axis=3)
        output = decode_labels(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])
        return output

    def predict_color_and_id_feature(self, feature):
        #self.cfg.INFER_SIZE[0:2] = feature.shape[1:3]
        #output_cmb = self.sess.run([self.output_from_feature, self.output_id_from_feature], feed_dict={self.logits_placeholder: feature})
        return self.sess.run([self.output_from_feature, self.output_id_from_feature], feed_dict={self.logits_placeholder: feature})


    def setup(self):
        arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=self.cfg.is_training)
        with slim.arg_scope(arg_scope):
            _, end_points = mobilenet_v1.mobilenet_v1(self.layers['data'], is_training=self.cfg.is_training, spatial_squeeze=False)

        for key in end_points.keys():
            if('Logits' not in key and 'predictions' not in key):
                self.layers[key] = end_points[key]
            
        
        (self.feed('Conv2d_13_pointwise')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_2_skip'))


        shape = self.layers['conv5_2_skip'].get_shape().as_list()[1:3]
        h, w = shape

        (self.feed('conv5_2_skip')
            .avg_pool(h, w, h, w, name='conv5_2_pool1')
            .conv(1, 1, 512//4, 1, 1, biased=False, relu=False, name='conv5_2_pool1_conv')
            .batch_normalization(relu=True, name='conv5_2_pool1_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool1_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/2, w/2, h/2, w/2, name='conv5_2_pool2')
            .conv(1, 1, 512//4, 1, 1, biased=False, relu=False, name='conv5_2_pool2_conv')
            .batch_normalization(relu=True, name='conv5_2_pool2_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool2_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/4, w/4, h/4, w/4, name='conv5_2_pool4')
            .conv(1, 1, 512//4, 1, 1, biased=False, relu=False, name='conv5_2_pool4_conv')
            .batch_normalization(relu=True, name='conv5_2_pool4_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool4_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/8, w/8, h/8, w/8, name='conv5_2_pool8')
            .conv(1, 1, 512//4, 1, 1, biased=False, relu=False, name='conv5_2_pool8_conv')
            .batch_normalization(relu=True, name='conv5_2_pool8_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool8_interp'))
        

        (self.feed('conv5_2_skip',
                   'conv5_2_pool8_interp',
                   'conv5_2_pool4_interp',
                   'conv5_2_pool2_interp',
                   'conv5_2_pool1_interp')
             .concat(axis=-1, name='conv5_2_concat')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_3')
             .batch_normalization(relu=True, name='conv5_3_bn'))


        # UpSample 1
        (self.feed('Conv2d_11_pointwise')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv4_2_skip_conv')
             .batch_normalization(relu=True, name='conv4_2_skip_conv_bn'))
        
        (self.feed('conv5_3_bn',
                   'conv4_2_skip_conv_bn')
             .add(name='up_4_add')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, padding='SAME', name='up_4')
             .batch_normalization(relu=True, name='up_4_bn'))
        

        # UpSample 2
        (self.feed('Conv2d_5_pointwise')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, padding='SAME', name='conv3_2_skip_conv')
             .batch_normalization(relu=True, name='conv3_2_skip_conv_bn'))

        (self.feed('up_4_bn',
                   'conv3_2_skip_conv_bn')
             .add(name='up_3_add')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='up_3')
             .batch_normalization(relu=True, name='up_3_bn'))
        

        # UpSample 3
        (self.feed('Conv2d_3_pointwise')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv2_2_skip_conv')
             .batch_normalization(relu=True, name='conv2_2_skip_conv_bn'))

        (self.feed('up_3_bn',
                   'conv2_2_skip_conv_bn')
             .add(name='up_2_add')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='up_2')
             .batch_normalization(relu=True, name='up_2_bn'))


        # Results
        (self.feed('up_2_bn')
             .conv(3, 3, self.cfg.param['num_classes'], 1, 1, biased=False, relu=False, padding='SAME', name='conv6_cls'))

class Swift_mobile_light(Network):
    def __init__(self, cfg, mode, image_reader=None, next_image = None, next_label = None):
        self.cfg = cfg
        self.mode = mode
        '''
        if mode == 'train':
            self.images, self.labels = image_reader.next_image, image_reader.next_label    
        
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)
        '''
        if mode == 'train':
            self.images, self.labels = next_image, next_label    
        
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)

        elif mode == 'eval':
            self.images, self.labels = image_reader.next_image, image_reader.next_label    
        
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)
            
            self.output = self.get_output_node()

        elif mode == 'inference':
            # Create placeholder and pre-process here.
            self.img_placeholder = tf.placeholder(dtype=tf.float32, shape=cfg.INFER_SIZE)
            self.logits_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.cfg.param['num_classes']])
            self.images, self.o_shape, self.n_shape = _infer_preprocess(self.img_placeholder, cfg.IMG_MEAN)
            
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)

            self.output = self.get_output_node()
            self.output_id = self.get_output_id()

            self.output_feature = self.layers['conv6_cls']

            self.output_from_feature = self.get_output_node_from_feature()
            self.output_id_from_feature = self.get_output_id_from_feature()

    def get_output_node(self):
        if self.mode == 'inference':
            # Get logits from final layer
            logits = self.layers['conv6_cls']

            # Upscale the logits and decode prediction to get final result.
            logits_up = tf.image.resize_bilinear(logits, size=self.o_shape, align_corners=True)
            #logits_up = tf.image.crop_to_bounding_box(logits_up, 0, 0, self.o_shape[0], self.o_shape[1])

            output_classes = tf.argmax(logits_up, axis=3)
            output = decode_labels(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])

        elif self.mode == 'eval':
            logits = self.layers['conv6_cls']

            logits_up = tf.image.resize_bilinear(logits, size=tf.shape(self.labels)[1:3], align_corners=True)
            output = tf.argmax(logits_up, axis=3)
            output = tf.expand_dims(output, axis=3)

        return output
    
    def get_output_id(self):
        if self.mode == 'inference':
            # Get logits from final layer
            logits = self.layers['conv6_cls']

            # Upscale the logits and decode prediction to get final result.
            logits_up = tf.image.resize_bilinear(logits, size=self.o_shape, align_corners=True)
            #logits_up = tf.image.crop_to_bounding_box(logits_up, 0, 0, self.o_shape[0], self.o_shape[1])

            output_classes = tf.argmax(logits_up, axis=3)
            output = decode_ids(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])

        elif self.mode == 'eval':
            logits = self.layers['conv6_cls']

            logits_up = tf.image.resize_bilinear(logits, size=tf.shape(self.labels)[1:3], align_corners=True)
            output = tf.argmax(logits_up, axis=3)
            output = tf.expand_dims(output, axis=3)

        return output

    def predict_feature(self, image, options=None, run_metadata=None):
        # The output layer
        return self.sess.run(self.output_feature, feed_dict = {self.img_placeholder: image}, options=options, run_metadata=run_metadata)

    def predict(self, image):
        return self.sess.run(self.output, feed_dict={self.img_placeholder: image})
    
    def predict_id(self, image):
        return self.sess.run(self.output_id, feed_dict={self.img_placeholder: image})
    
    def predict_color_and_id(self, image):
        #output_cmb = self.sess.run([self.output, self.output_id], feed_dict={self.img_placeholder: image})
        return self.sess.run([self.output, self.output_id], feed_dict={self.img_placeholder: image})
    
    def get_output_id_from_feature(self):
        # Upscale the logits and decode prediction to get final result.
        logits_up = tf.image.resize_bilinear(self.logits_placeholder, size=self.o_shape, align_corners=True)
        #logits_up = tf.image.crop_to_bounding_box(logits_up, 0, 0, self.o_shape[0], self.o_shape[1])

        output_classes = tf.argmax(logits_up, axis=3)
        output_id = decode_ids(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])
        return output_id
    
    def get_output_node_from_feature(self):
        # Upscale the logits and decode prediction to get final result.
        logits_up = tf.image.resize_bilinear(self.logits_placeholder, size=self.o_shape, align_corners=True)
        #logits_up = tf.image.crop_to_bounding_box(logits_up, 0, 0, self.o_shape[0], self.o_shape[1])

        output_classes = tf.argmax(logits_up, axis=3)
        output = decode_labels(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])
        return output

    def predict_color_and_id_feature(self, feature):
        #self.cfg.INFER_SIZE[0:2] = feature.shape[1:3]
        #output_cmb = self.sess.run([self.output_from_feature, self.output_id_from_feature], feed_dict={self.logits_placeholder: feature})
        return self.sess.run([self.output_from_feature, self.output_id_from_feature], feed_dict={self.logits_placeholder: feature})


    def setup(self):
        arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=self.cfg.is_training)
        with slim.arg_scope(arg_scope):
            _, end_points = mobilenet_v1.mobilenet_v1(self.layers['data'], is_training=self.cfg.is_training, spatial_squeeze=False)

        for key in end_points.keys():
            if('Logits' not in key and 'predictions' not in key):
                self.layers[key] = end_points[key]
            
        
        (self.feed('Conv2d_13_pointwise')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_2_skip'))


        shape = self.layers['conv5_2_skip'].get_shape().as_list()[1:3]
        h, w = shape

        (self.feed('conv5_2_skip')
            .avg_pool(h, w, h, w, name='conv5_2_pool1')
            .conv(1, 1, 512//4, 1, 1, biased=False, relu=False, name='conv5_2_pool1_conv')
            .batch_normalization(relu=True, name='conv5_2_pool1_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool1_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/2, w/2, h/2, w/2, name='conv5_2_pool2')
            .conv(1, 1, 512//4, 1, 1, biased=False, relu=False, name='conv5_2_pool2_conv')
            .batch_normalization(relu=True, name='conv5_2_pool2_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool2_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/4, w/4, h/4, w/4, name='conv5_2_pool4')
            .conv(1, 1, 512//4, 1, 1, biased=False, relu=False, name='conv5_2_pool4_conv')
            .batch_normalization(relu=True, name='conv5_2_pool4_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool4_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/8, w/8, h/8, w/8, name='conv5_2_pool8')
            .conv(1, 1, 512//4, 1, 1, biased=False, relu=False, name='conv5_2_pool8_conv')
            .batch_normalization(relu=True, name='conv5_2_pool8_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool8_interp'))
        

        (self.feed('conv5_2_skip',
                   'conv5_2_pool8_interp',
                   'conv5_2_pool4_interp',
                   'conv5_2_pool2_interp',
                   'conv5_2_pool1_interp')
             .concat(axis=-1, name='conv5_2_concat')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_3')
             .batch_normalization(relu=True, name='conv5_3_bn'))


        # UpSample 1
        (self.feed('Conv2d_11_pointwise')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, padding='SAME', name='conv4_2_skip_conv')
             .batch_normalization(relu=True, name='conv4_2_skip_conv_bn'))
        
        (self.feed('conv5_3_bn',
                   'conv4_2_skip_conv_bn')
             .add(name='up_4_add')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='up_4')
             .batch_normalization(relu=True, name='up_4_bn'))
        

        # UpSample 2
        (self.feed('Conv2d_5_pointwise')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv3_2_skip_conv')
             .batch_normalization(relu=True, name='conv3_2_skip_conv_bn'))

        (self.feed('up_4_bn',
                   'conv3_2_skip_conv_bn')
             .add(name='up_3_add')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='up_3')
             .batch_normalization(relu=True, name='up_3_bn'))
        

        # UpSample 3
        (self.feed('Conv2d_3_pointwise')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv2_2_skip_conv')
             .batch_normalization(relu=True, name='conv2_2_skip_conv_bn'))

        (self.feed('up_3_bn',
                   'conv2_2_skip_conv_bn')
             .add(name='up_2_add')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='up_2')
             .batch_normalization(relu=True, name='up_2_bn'))


        # Results
        (self.feed('up_2_bn')
             .conv(3, 3, self.cfg.param['num_classes'], 1, 1, biased=False, relu=False, padding='SAME', name='conv6_cls'))

# ---------------------------- Res18_bn --------------------------------#
class res18_bn(Swift_mobile_light):
    def setup(self):
        # Double Conv_3*3 version of ResNet18
        (self.feed('data')
             .conv(3, 3, 32, 2, 2, biased=False, relu=False, padding='SAME', name='conv1_1_3x3')
             .batch_normalization(relu=True, name='conv1_1_3x3_bn')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_2_3x3')
             .batch_normalization(relu=True, name='conv1_2_3x3_bn')
             .max_pool(3, 3, 2, 2, padding='SAME', name='pool1_3x3_s2'))
        

        (self.feed('pool1_3x3_s2')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv2_1_3x3_reduce')
             .batch_normalization(relu=True, name='conv2_1_3x3_reduce_bn')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv2_1_3x3')
             .batch_normalization(relu=True, name='conv2_1_3x3_bn'))

        (self.feed('pool1_3x3_s2',
                   'conv2_1_3x3_bn')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv2_2_3x3_reduce')
             .batch_normalization(relu=True, name='conv2_2_3x3_reduce_bn')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv2_2_3x3')
             .batch_normalization(relu=True, name='conv2_2_3x3_bn'))


        (self.feed('conv2_1/relu',
                   'conv2_2_3x3_bn')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, padding='SAME', name='conv3_1_1x1_proj')
             .batch_normalization(relu=True, name='conv3_1_1x1_proj_bn'))

        (self.feed('conv2_2/relu')
             .conv(3, 3, 128, 2, 2, biased=False, relu=False, padding='SAME', name='conv3_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv3_1_3x3')
             .batch_normalization(relu=True, name='conv3_1_3x3_bn'))

        (self.feed('conv3_1_1x1_proj_bn',
                   'conv3_1_3x3_bn')
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv3_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv3_2_3x3')
             .batch_normalization(relu=True, name='conv3_2_3x3_bn'))


        (self.feed('conv3_1/relu',
                   'conv3_2_3x3_bn')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 256, 2, 2, biased=False, relu=False, padding='SAME', name='conv4_1_1x1_proj')
             .batch_normalization(relu=False, name='conv4_1_1x1_proj_bn'))

        (self.feed('conv3_2/relu')
             .conv(3, 3, 256, 2, 2, biased=False, relu=False, padding='SAME', name='conv4_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, padding='SAME', name='conv4_1_3x3')
             .batch_normalization(relu=True, name='conv4_1_3x3_bn'))

        (self.feed('conv4_1_1x1_proj_bn',
                   'conv4_1_3x3_bn')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, padding='SAME', name='conv4_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
             .conv(3, 3, 256, 1, 1, biased=False, relu=False, padding='SAME', name='conv4_2_3x3')
             .batch_normalization(relu=True, name='conv4_2_3x3_bn'))

        (self.feed('conv4_1/relu',
                   'conv4_2_3x3_bn')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, padding='SAME', name='conv5_1_1x1_proj')
             .batch_normalization(relu=False, name='conv5_1_1x1_proj_bn'))

        (self.feed('conv4_2/relu')
             .conv(3, 3, 512, 2, 2, biased=False, relu=False, padding='SAME', name='conv5_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_1_3x3')
             .batch_normalization(relu=True, name='conv5_1_3x3_bn'))

        (self.feed('conv5_1_1x1_proj_bn',
                   'conv5_1_3x3_bn')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_2_3x3')
             .batch_normalization(relu=True, name='conv5_2_3x3_bn'))

        (self.feed('conv5_1/relu',
                   'conv5_2_3x3_bn')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_2/skip'))


        shape = self.layers['conv5_2/skip'].get_shape().as_list()[1:3]
        h, w = shape

        (self.feed('conv5_2/skip')
            .avg_pool(h, w, h, w, name='conv5_2_pool1')
            .batch_normalization(relu=True, name='conv5_2_pool1_conv_bn')
            .conv(1, 1, 512//4, 1, 1, biased=False, relu=False, name='conv5_2_pool1_conv')
            .resize_bilinear(shape, name='conv5_2_pool1_interp'))

        (self.feed('conv5_2/skip')
            .avg_pool(h/2, w/2, h/2, w/2, name='conv5_2_pool2')
            .batch_normalization(relu=True, name='conv5_2_pool2_conv_bn')
            .conv(1, 1, 512//4, 1, 1, biased=False, relu=False, name='conv5_2_pool2_conv')
            .resize_bilinear(shape, name='conv5_2_pool2_interp'))

        (self.feed('conv5_2/skip')
            .avg_pool(h/4, w/4, h/4, w/4, name='conv5_2_pool4')
            .batch_normalization(relu=True, name='conv5_2_pool3_conv_bn')
            .conv(1, 1, 512//4, 1, 1, biased=False, relu=False, name='conv5_2_pool4_conv')
            .resize_bilinear(shape, name='conv5_2_pool4_interp'))

        (self.feed('conv5_2/skip')
            .avg_pool(h/8, w/8, h/8, w/8, name='conv5_2_pool8')
            .batch_normalization(relu=True, name='conv5_2_pool6_conv_bn')
            .conv(1, 1, 512//4, 1, 1, biased=False, relu=False, name='conv5_2_pool8_conv')
            .resize_bilinear(shape, name='conv5_2_pool8_interp'))
        

        (self.feed('conv5_2/skip',
                   'conv5_2_pool8_interp',
                   'conv5_2_pool4_interp',
                   'conv5_2_pool2_interp',
                   'conv5_2_pool1_interp')
             .concat(axis=-1, name='conv5_2_concat')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_3')
             .batch_normalization(relu=True, name='conv5_3_bn'))
        

        # UpSample 1
        (self.feed('conv4_2')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, padding='SAME', name='conv4_2_skip_conv')
             .batch_normalization(relu=True, name='conv4_2_skip_conv_bn'))
        
        (self.feed('conv5_3_bn',
                   'conv4_2_skip_conv_bn')
             .add(name='up_4_add')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='up_4')
             .batch_normalization(relu=True, name='up_4_bn'))
        

        # UpSample 2
        (self.feed('conv3_2')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv3_2_skip_conv')
             .batch_normalization(relu=True, name='conv3_2_skip_conv_bn'))

        (self.feed('up_4_bn',
                   'conv3_2_skip_conv_bn')
             .add(name='up_3_add')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='up_3')
             .batch_normalization(relu=True, name='up_3_bn'))
        

        # UpSample 3
        (self.feed('conv2_2')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv2_2_skip_conv')
             .batch_normalization(relu=True, name='conv2_2_skip_conv_bn'))

        (self.feed('up_3_bn',
                   'conv2_2_skip_conv_bn')
             .add(name='up_2_add')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='up_2')
             .batch_normalization(relu=True, name='up_2_bn'))


        # Results
        (self.feed('up_2_bn')
             .conv(3, 3, self.cfg.param['num_classes'], 1, 1, biased=False, relu=False, padding='SAME', name='conv6_cls'))


# ---------------------- mobile_050 freetech ---------------------------#
class freetech_mobile_050(Network):
    def __init__(self, cfg, mode, image_reader=None, next_image = None, next_label = None):
        self.cfg = cfg
        self.mode = mode
        
        if mode == 'train':
            self.images, self.labels = next_image, next_label    
        
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)

        elif mode == 'eval':
            self.images, self.labels = image_reader.next_image, image_reader.next_label    
        
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)
            
            self.output = self.get_output_node()

        elif mode == 'inference':
            # Create placeholder and pre-process here.
            self.img_placeholder = tf.placeholder(dtype=tf.float32, shape=cfg.INFER_SIZE)
            self.logits_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.cfg.param['num_classes']])
            self.images, self.o_shape, self.n_shape = _infer_preprocess(self.img_placeholder, cfg.IMG_MEAN)
            
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)

            self.output = self.get_output_node()
            self.output_id = self.get_output_id()

            self.output_feature = self.layers['conv6_cls']

            self.output_from_feature = self.get_output_node_from_feature()
            self.output_id_from_feature = self.get_output_id_from_feature()

    def get_output_node(self):
        if self.mode == 'inference':
            # Get logits from final layer
            logits = self.layers['conv6_cls']
            # Upscale the logits and decode prediction to get final result.
            logits_up = tf.image.resize_bilinear(logits, size=self.o_shape, align_corners=True)
            output_classes = tf.argmax(logits_up, axis=3)
            output = decode_labels(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])

        elif self.mode == 'eval':
            logits = self.layers['conv6_cls']
            logits_up = tf.image.resize_bilinear(logits, size=tf.shape(self.labels)[1:3], align_corners=True)
            output = tf.argmax(logits_up, axis=3)
            output = tf.expand_dims(output, axis=3)

        return output
    
    def get_output_id(self):
        if self.mode == 'inference':
            # Get logits from final layer
            logits = self.layers['conv6_cls']
            logits_up = tf.image.resize_bilinear(logits, size=self.o_shape, align_corners=True)
            output_classes = tf.argmax(logits_up, axis=3)
            output = decode_ids(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])

        elif self.mode == 'eval':
            logits = self.layers['conv6_cls']
            logits_up = tf.image.resize_bilinear(logits, size=tf.shape(self.labels)[1:3], align_corners=True)
            output = tf.argmax(logits_up, axis=3)
            output = tf.expand_dims(output, axis=3)

        return output

    def predict_feature(self, image):
        # The output layer
        return self.sess.run(self.output_feature, feed_dict = {self.img_placeholder: image})

    def predict(self, image):
        return self.sess.run(self.output, feed_dict={self.img_placeholder: image})
    
    def predict_id(self, image):
        return self.sess.run(self.output_id, feed_dict={self.img_placeholder: image})
    
    def predict_color_and_id(self, image):
        return self.sess.run([self.output, self.output_id], feed_dict={self.img_placeholder: image})
    
    def get_output_id_from_feature(self):
        # Upscale the logits and decode prediction to get final result.
        logits_up = tf.image.resize_bilinear(self.logits_placeholder, size=self.o_shape, align_corners=True)
        output_classes = tf.argmax(logits_up, axis=3)
        output_id = decode_ids(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])
        return output_id
    
    def get_output_node_from_feature(self):
        # Upscale the logits and decode prediction to get final result.
        logits_up = tf.image.resize_bilinear(self.logits_placeholder, size=self.o_shape, align_corners=True)
        output_classes = tf.argmax(logits_up, axis=3)
        output = decode_labels(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])
        return output

    def predict_color_and_id_feature(self, feature):
        return self.sess.run([self.output_from_feature, self.output_id_from_feature], feed_dict={self.logits_placeholder: feature})
    
    def setup(self):
        arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=self.cfg.is_training)
        with slim.arg_scope(arg_scope):
            _, end_points = mobilenet_v1.mobilenet_v1_050(self.layers['data'], is_training=self.cfg.is_training, spatial_squeeze=False)

        for key in end_points.keys():
            if('Logits' not in key and 'predictions' not in key):
                self.layers[key] = end_points[key]
            
        
        (self.feed('Conv2d_13_pointwise')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_2_skip'))


        shape = self.layers['conv5_2_skip'].get_shape().as_list()[1:3]
        h, w = shape

        (self.feed('conv5_2_skip')
            .avg_pool(h, w, h, w, name='conv5_2_pool1')
            .conv(1, 1, 256//4, 1, 1, biased=False, relu=False, name='conv5_2_pool1_conv')
            .batch_normalization(relu=True, name='conv5_2_pool1_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool1_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/2, w/2, h/2, w/2, name='conv5_2_pool2')
            .conv(1, 1, 256//4, 1, 1, biased=False, relu=False, name='conv5_2_pool2_conv')
            .batch_normalization(relu=True, name='conv5_2_pool2_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool2_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/4, w/4, h/4, w/4, name='conv5_2_pool4')
            .conv(1, 1, 256//4, 1, 1, biased=False, relu=False, name='conv5_2_pool4_conv')
            .batch_normalization(relu=True, name='conv5_2_pool4_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool4_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/8, w/8, h/8, w/8, name='conv5_2_pool8')
            .conv(1, 1, 256//4, 1, 1, biased=False, relu=False, name='conv5_2_pool8_conv')
            .batch_normalization(relu=True, name='conv5_2_pool8_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool8_interp'))
        

        (self.feed('conv5_2_skip',
                   'conv5_2_pool8_interp',
                   'conv5_2_pool4_interp',
                   'conv5_2_pool2_interp',
                   'conv5_2_pool1_interp')
             .concat(axis=-1, name='conv5_2_concat')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_3')
             .batch_normalization(relu=True, name='conv5_3_bn'))

        # ------------------------------- Segmentation Branch --------------------------------------- #
        # UpSample 1
        (self.feed('Conv2d_11_pointwise')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv4_2_skip_conv')
             .batch_normalization(relu=True, name='conv4_2_skip_conv_bn'))
        
        (self.feed('conv5_3_bn',
                   'conv4_2_skip_conv_bn')
             .add(name='up_4_add')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='up_4')
             .batch_normalization(relu=True, name='up_4_bn'))
        

        # UpSample 2
        (self.feed('Conv2d_5_pointwise')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv3_2_skip_conv')
             .batch_normalization(relu=True, name='conv3_2_skip_conv_bn'))

        (self.feed('up_4_bn',
                   'conv3_2_skip_conv_bn')
             .add(name='up_3_add')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, padding='SAME', name='up_3')
             .batch_normalization(relu=True, name='up_3_bn'))
        

        # UpSample 3
        (self.feed('Conv2d_3_pointwise')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, padding='SAME', name='conv2_2_skip_conv')
             .batch_normalization(relu=True, name='conv2_2_skip_conv_bn'))

        (self.feed('up_3_bn',
                   'conv2_2_skip_conv_bn')
             .add(name='up_2_add')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, padding='SAME', name='up_2')
             .batch_normalization(relu=True, name='up_2_bn'))


        # Results

        (self.feed('up_4_bn')
             .conv(3, 3, self.cfg.param['num_classes'], 1, 1, biased=False, relu=False, padding='SAME', name='conv4_cls'))

        (self.feed('up_3_bn')
             .conv(3, 3, self.cfg.param['num_classes'], 1, 1, biased=False, relu=False, padding='SAME', name='conv5_cls'))

        (self.feed('up_2_bn')
             .conv(3, 3, self.cfg.param['num_classes'], 1, 1, biased=False, relu=False, padding='SAME', name='conv6_cls'))


# Lane and segmentation
class freetech_mobile_050_lane(Network):

    def __init__(self, cfg, mode, image_reader=None, next_image = None, next_label = None):
        self.cfg = cfg
        self.mode = mode
        
        if mode == 'train':
            self.images, self.labels = next_image, next_label    
        
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)

        elif mode == 'eval':
            self.images, self.labels = image_reader.next_image, image_reader.next_label    
        
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)
            
            self.output = self.get_output_node()

        elif mode == 'inference':
            # Create placeholder and pre-process here.
            self.img_placeholder = tf.placeholder(dtype=tf.float32, shape=cfg.INFER_SIZE)
            self.logits_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, None, self.cfg.param['num_classes']])
            self.images, self.o_shape, self.n_shape = _infer_preprocess(self.img_placeholder, cfg.IMG_MEAN)
            
            super().__init__(inputs={'data': self.images}, cfg=self.cfg)

            self.output = self.get_output_node()
            self.output_id = self.get_output_id()

            self.output_lane = self.get_output_node(is_lane=True)
            self.output_id_lane = self.get_output_id(is_lane=True)

            self.output_feature = self.layers['conv6_cls']
    
    def get_output_node(self, is_lane=False):
        if self.mode == 'inference':
            # Get logits from final layer
            if is_lane:
                logits = self.layers['lane/conv6_cls']
            else:
                logits = self.layers['conv6_cls']
            # Upscale the logits and decode prediction to get final result.
            logits_up = tf.image.resize_bilinear(logits, size=self.o_shape, align_corners=True)
            output_classes = tf.argmax(logits_up, axis=3)

            if is_lane:
                output = decode_labels(output_classes, self.o_shape[0:2], 2)
            else:
                output = decode_labels(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])

        elif self.mode == 'eval':
            logits = self.layers['conv6_cls']
            logits_up = tf.image.resize_bilinear(logits, size=tf.shape(self.labels)[1:3], align_corners=True)
            output = tf.argmax(logits_up, axis=3)
            output = tf.expand_dims(output, axis=3)

        return output
    
    def get_output_id(self, is_lane=False):
        if self.mode == 'inference':
            # Get logits from final layer
            if is_lane:
                logits = self.layers['lane/conv6_cls']
            else:
                logits = self.layers['conv6_cls']
            # Upscale the logits and decode prediction to get final result.
            logits_up = tf.image.resize_bilinear(logits, size=self.o_shape, align_corners=True)
            output_classes = tf.argmax(logits_up, axis=3)
            if is_lane:
                output = decode_ids(output_classes, self.o_shape[0:2], 2)
            else:
                output = decode_ids(output_classes, self.o_shape[0:2], self.cfg.param['num_classes'])

        elif self.mode == 'eval':
            logits = self.layers['conv6_cls']
            logits_up = tf.image.resize_bilinear(logits, size=tf.shape(self.labels)[1:3], align_corners=True)
            output = tf.argmax(logits_up, axis=3)
            output = tf.expand_dims(output, axis=3)

        return output

    def predict_color_and_id_seg_lane(self, image):
        return self.sess.run([self.output, self.output_id, self.output_lane, self.output_id_lane], feed_dict={self.img_placeholder: image})
    
    def setup(self):
        arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=self.cfg.is_training)
        with slim.arg_scope(arg_scope):
            _, end_points = mobilenet_v1.mobilenet_v1_050(self.layers['data'], is_training=self.cfg.is_training, spatial_squeeze=False)

        for key in end_points.keys():
            if('Logits' not in key and 'predictions' not in key):
                self.layers[key] = end_points[key]
            
        
        (self.feed('Conv2d_13_pointwise')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_2_skip'))


        shape = self.layers['conv5_2_skip'].get_shape().as_list()[1:3]
        h, w = shape

        (self.feed('conv5_2_skip')
            .avg_pool(h, w, h, w, name='conv5_2_pool1')
            .conv(1, 1, 256//4, 1, 1, biased=False, relu=False, name='conv5_2_pool1_conv')
            .batch_normalization(relu=True, name='conv5_2_pool1_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool1_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/2, w/2, h/2, w/2, name='conv5_2_pool2')
            .conv(1, 1, 256//4, 1, 1, biased=False, relu=False, name='conv5_2_pool2_conv')
            .batch_normalization(relu=True, name='conv5_2_pool2_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool2_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/4, w/4, h/4, w/4, name='conv5_2_pool4')
            .conv(1, 1, 256//4, 1, 1, biased=False, relu=False, name='conv5_2_pool4_conv')
            .batch_normalization(relu=True, name='conv5_2_pool4_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool4_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/8, w/8, h/8, w/8, name='conv5_2_pool8')
            .conv(1, 1, 256//4, 1, 1, biased=False, relu=False, name='conv5_2_pool8_conv')
            .batch_normalization(relu=True, name='conv5_2_pool8_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool8_interp'))
        

        (self.feed('conv5_2_skip',
                   'conv5_2_pool8_interp',
                   'conv5_2_pool4_interp',
                   'conv5_2_pool2_interp',
                   'conv5_2_pool1_interp')
             .concat(axis=-1, name='conv5_2_concat')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_3')
             .batch_normalization(relu=True, name='conv5_3_bn'))

        # ------------------------------- Segmentation Branch --------------------------------------- #
        # UpSample 1
        (self.feed('Conv2d_11_pointwise')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv4_2_skip_conv')
             .batch_normalization(relu=True, name='conv4_2_skip_conv_bn'))
        
        (self.feed('conv5_3_bn',
                   'conv4_2_skip_conv_bn')
             .add(name='up_4_add')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='up_4')
             .batch_normalization(relu=True, name='up_4_bn'))
        

        # UpSample 2
        (self.feed('Conv2d_5_pointwise')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv3_2_skip_conv')
             .batch_normalization(relu=True, name='conv3_2_skip_conv_bn'))

        (self.feed('up_4_bn',
                   'conv3_2_skip_conv_bn')
             .add(name='up_3_add')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, padding='SAME', name='up_3')
             .batch_normalization(relu=True, name='up_3_bn'))
        

        # UpSample 3
        (self.feed('Conv2d_3_pointwise')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, padding='SAME', name='conv2_2_skip_conv')
             .batch_normalization(relu=True, name='conv2_2_skip_conv_bn'))

        (self.feed('up_3_bn',
                   'conv2_2_skip_conv_bn')
             .add(name='up_2_add')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, padding='SAME', name='up_2')
             .batch_normalization(relu=True, name='up_2_bn'))


        # Results

        (self.feed('up_4_bn')
             .conv(3, 3, self.cfg.param['num_classes'], 1, 1, biased=False, relu=False, padding='SAME', name='conv4_cls'))

        (self.feed('up_3_bn')
             .conv(3, 3, self.cfg.param['num_classes'], 1, 1, biased=False, relu=False, padding='SAME', name='conv5_cls'))

        (self.feed('up_2_bn')
             .conv(3, 3, self.cfg.param['num_classes'], 1, 1, biased=False, relu=False, padding='SAME', name='conv6_cls'))

        
        # ------------------------------------- Lane Branch --------------------------------------- #
        # UpSample 1
        (self.feed('Conv2d_11_pointwise')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv4_2_skip_conv')
             .batch_normalization(relu=True, name='lane/conv4_2_skip_conv_bn'))
        
        # conv5_3  align
        (self.feed('conv5_3_bn')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv5_3')
             .batch_normalization(relu=True, name='lane/conv5_3_bn'))

        (self.feed('lane/conv5_3_bn',
                   'lane/conv4_2_skip_conv_bn')
             .add(name='lane/up_4_add')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, padding='SAME', name='lane/up_4')
             .batch_normalization(relu=True, name='lane/up_4_bn'))
        

        # UpSample 2
        (self.feed('Conv2d_5_pointwise')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv3_2_skip_conv')
             .batch_normalization(relu=True, name='lane/conv3_2_skip_conv_bn'))

        (self.feed('lane/up_4_bn',
                   'lane/conv3_2_skip_conv_bn')
             .add(name='lane/up_3_add')
             .conv(3, 3, 16, 1, 1, biased=False, relu=False, padding='SAME', name='lane/up_3')
             .batch_normalization(relu=True, name='lane/up_3_bn'))
        

        # UpSample 3
        (self.feed('Conv2d_3_pointwise')
             .conv(1, 1, 16, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv2_2_skip_conv')
             .batch_normalization(relu=True, name='lane/conv2_2_skip_conv_bn'))

        (self.feed('lane/up_3_bn',
                   'lane/conv2_2_skip_conv_bn')
             .add(name='lane/up_2_add')
             .conv(3, 3, 16, 1, 1, biased=False, relu=False, padding='SAME', name='lane/up_2')
             .batch_normalization(relu=True, name='lane/up_2_bn'))


        # Results

        (self.feed('lane/up_4_bn')
             .conv(3, 3, 2, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv4_cls'))

        (self.feed('lane/up_3_bn')
             .conv(3, 3, 2, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv5_cls'))

        (self.feed('lane/up_2_bn')
             .conv(3, 3, 2, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv6_cls'))

# mobile 025  & knowledge distillation
class freetech_mobile_025_lane(freetech_mobile_050_lane):
    def setup(self):
        arg_scope = mobilenet_v1.mobilenet_v1_arg_scope(is_training=self.cfg.is_training)
        with slim.arg_scope(arg_scope):
            _, end_points = mobilenet_v1.mobilenet_v1_025(self.layers['data'], is_training=self.cfg.is_training, spatial_squeeze=False)

        for key in end_points.keys():
            if('Logits' not in key and 'predictions' not in key):
                self.layers[key] = end_points[key]
            
        
        (self.feed('Conv2d_13_pointwise')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_2_skip'))


        shape = self.layers['conv5_2_skip'].get_shape().as_list()[1:3]
        h, w = shape

        (self.feed('conv5_2_skip')
            .avg_pool(h, w, h, w, name='conv5_2_pool1')
            .conv(1, 1, 128//4, 1, 1, biased=False, relu=False, name='conv5_2_pool1_conv')
            .batch_normalization(relu=True, name='conv5_2_pool1_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool1_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/2, w/2, h/2, w/2, name='conv5_2_pool2')
            .conv(1, 1, 128//4, 1, 1, biased=False, relu=False, name='conv5_2_pool2_conv')
            .batch_normalization(relu=True, name='conv5_2_pool2_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool2_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/4, w/4, h/4, w/4, name='conv5_2_pool4')
            .conv(1, 1, 128//4, 1, 1, biased=False, relu=False, name='conv5_2_pool4_conv')
            .batch_normalization(relu=True, name='conv5_2_pool4_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool4_interp'))

        (self.feed('conv5_2_skip')
            .avg_pool(h/8, w/8, h/8, w/8, name='conv5_2_pool8')
            .conv(1, 1, 128//4, 1, 1, biased=False, relu=False, name='conv5_2_pool8_conv')
            .batch_normalization(relu=True, name='conv5_2_pool8_conv_bn')
            .resize_bilinear(shape, name='conv5_2_pool8_interp'))
        

        (self.feed('conv5_2_skip',
                   'conv5_2_pool8_interp',
                   'conv5_2_pool4_interp',
                   'conv5_2_pool2_interp',
                   'conv5_2_pool1_interp')
             .concat(axis=-1, name='conv5_2_concat')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_3')
             .batch_normalization(relu=True, name='conv5_3_bn'))

        # ------------------------------- Segmentation Branch --------------------------------------- #
        # UpSample 1
        (self.feed('Conv2d_11_pointwise')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv4_2_skip_conv')
             .batch_normalization(relu=True, name='conv4_2_skip_conv_bn'))
        
        (self.feed('conv5_3_bn',
                   'conv4_2_skip_conv_bn')
             .add(name='up_4_add')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='up_4')
             .batch_normalization(relu=True, name='up_4_bn'))
        

        # UpSample 2
        (self.feed('Conv2d_5_pointwise')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv3_2_skip_conv')
             .batch_normalization(relu=True, name='conv3_2_skip_conv_bn'))

        (self.feed('up_4_bn',
                   'conv3_2_skip_conv_bn')
             .add(name='up_3_add')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, padding='SAME', name='up_3')
             .batch_normalization(relu=True, name='up_3_bn'))
        

        # UpSample 3
        (self.feed('Conv2d_3_pointwise')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, padding='SAME', name='conv2_2_skip_conv')
             .batch_normalization(relu=True, name='conv2_2_skip_conv_bn'))

        (self.feed('up_3_bn',
                   'conv2_2_skip_conv_bn')
             .add(name='up_2_add')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, padding='SAME', name='up_2')
             .batch_normalization(relu=True, name='up_2_bn'))


        # Results

        (self.feed('up_4_bn')
             .conv(3, 3, self.cfg.param['num_classes'], 1, 1, biased=False, relu=False, padding='SAME', name='conv4_cls'))

        (self.feed('up_3_bn')
             .conv(3, 3, self.cfg.param['num_classes'], 1, 1, biased=False, relu=False, padding='SAME', name='conv5_cls'))

        (self.feed('up_2_bn')
             .conv(3, 3, self.cfg.param['num_classes'], 1, 1, biased=False, relu=False, padding='SAME', name='conv6_cls'))

        
        # ------------------------------------- Lane Branch --------------------------------------- #
        # UpSample 1
        (self.feed('Conv2d_11_pointwise')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv4_2_skip_conv')
             .batch_normalization(relu=True, name='lane/conv4_2_skip_conv_bn'))
        
        # conv5_3  align
        (self.feed('conv5_3_bn')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv5_3')
             .batch_normalization(relu=True, name='lane/conv5_3_bn'))

        (self.feed('lane/conv5_3_bn',
                   'lane/conv4_2_skip_conv_bn')
             .add(name='lane/up_4_add')
             .conv(3, 3, 32, 1, 1, biased=False, relu=False, padding='SAME', name='lane/up_4')
             .batch_normalization(relu=True, name='lane/up_4_bn'))
        

        # UpSample 2
        (self.feed('Conv2d_5_pointwise')
             .conv(1, 1, 32, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv3_2_skip_conv')
             .batch_normalization(relu=True, name='lane/conv3_2_skip_conv_bn'))

        (self.feed('lane/up_4_bn',
                   'lane/conv3_2_skip_conv_bn')
             .add(name='lane/up_3_add')
             .conv(3, 3, 16, 1, 1, biased=False, relu=False, padding='SAME', name='lane/up_3')
             .batch_normalization(relu=True, name='lane/up_3_bn'))
        

        # UpSample 3
        (self.feed('Conv2d_3_pointwise')
             .conv(1, 1, 16, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv2_2_skip_conv')
             .batch_normalization(relu=True, name='lane/conv2_2_skip_conv_bn'))

        (self.feed('lane/up_3_bn',
                   'lane/conv2_2_skip_conv_bn')
             .add(name='lane/up_2_add')
             .conv(3, 3, 16, 1, 1, biased=False, relu=False, padding='SAME', name='lane/up_2')
             .batch_normalization(relu=True, name='lane/up_2_bn'))


        # Results

        (self.feed('lane/up_4_bn')
             .conv(3, 3, 2, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv4_cls'))

        (self.feed('lane/up_3_bn')
             .conv(3, 3, 2, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv5_cls'))

        (self.feed('lane/up_2_bn')
             .conv(3, 3, 2, 1, 1, biased=False, relu=False, padding='SAME', name='lane/conv6_cls'))