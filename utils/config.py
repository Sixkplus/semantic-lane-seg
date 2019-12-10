import numpy as np
import os

class Config(object):
    # Setting dataset directory
    # CITYSCAPES_DATA_DIR = './data/cityscapes_dataset/cityscape/'
    CITYSCAPES_DATA_DIR = './data/input/cityscapes'
    ADE20K_DATA_DIR = './data/ADEChallengeData2016/'
    
    
    ADE20K_eval_list = os.path.join('./data/list/ade20k_val_list.txt')
    CITYSCAPES_eval_list = os.path.join('./data/list/cityscapes_val_list.txt')
    CITYSCAPES_motion_eval_list = os.path.join('./data/list/cityscapes_motion_train_list.txt')
    
    
    ADE20K_train_list = os.path.join('./data/list/ade20k_train_list.txt')
    CITYSCAPES_train_list = os.path.join('./data/list/cityscapes_train_list.txt')
    CITYSCAPES_motion_train_list = os.path.join('./data/list/cityscapes_motion_train_list.txt')
    

    # Camvid
    CAMVID_DATA_DIR = './data/input/camvid'
    camvid_train_list = os.path.join('./data/list/camvid_train_list.txt')
    camvid_eval_list = os.path.join('./data/list/camvid_val_list.txt')

    # Freetech
    FREETECH_DATA_DIR = './data/input/freetech'
    freetech_train_list = os.path.join('./data/list/freetech_train_list.txt')
    freetech_eval_list = os.path.join('./data/list/freetech_val_list.txt')

    # Freetech lane
    FREETECH_LANE_DIR = './data/input/freetech'
    freetech_lane_train_list = os.path.join('./data/list/freetech_lane_train_list.txt')
    freetech_lane_eval_list = os.path.join('./data/list/freetech_lane_val_list.txt')

    # Agriculture
    AGRIC_DATA_DIR = './data/agriculture'
    agric_train_list = os.path.join('./data/list/agriculture_train_list.txt')
    agric_eval_list = os.path.join('./data/list/agriculture_train_list.txt')
    
    IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
    
    ADE20k_param = {'name': 'ade20k',
                'num_classes': 150, # predict: [0~149] corresponding to label [1~150], ignore class 0 (background) 
                'ignore_label': 0,
                'eval_size': [480, 480],
                'eval_steps': 2000,
                'eval_list': ADE20K_eval_list,
                'train_list': ADE20K_train_list,
                'data_dir': ADE20K_DATA_DIR}
    
    cityscapes_param = {'name': 'cityscapes',
                    'num_classes': 19,
                    'ignore_label': 255,
                    'eval_size': [1024, 2048],
                    'eval_steps': 500,
                    'eval_list': CITYSCAPES_eval_list,
                    'train_list': CITYSCAPES_train_list,
                    'data_dir': CITYSCAPES_DATA_DIR}
    
    cityscapes_motion_param = {'name': 'cityscapes_motion',
                    'num_classes': 19,
                    'ignore_label': 255,
                    'eval_size': [1024, 2048],
                    'eval_steps': 500,
                    'eval_list': CITYSCAPES_motion_eval_list,
                    'train_list': CITYSCAPES_motion_train_list,
                    'data_dir': CITYSCAPES_DATA_DIR}
    
    freetech_param = {'name': 'freetech',
                    'num_classes': 6,
                    'ignore_label': 255,
                    'eval_size': [1256, 1928],
                    'eval_steps': 1200,
                    'eval_list': freetech_eval_list,
                    'train_list': freetech_train_list,
                    'data_dir': FREETECH_DATA_DIR}
    
    freetech_lane_param = {'name': 'freetech_lane',
                    'num_classes': 6,
                    'ignore_label': 255,
                    'eval_size': [1256, 1928],
                    'eval_steps': 1021,
                    'eval_list': freetech_lane_eval_list,
                    'train_list': freetech_lane_train_list,
                    'data_dir': FREETECH_LANE_DIR}
    
    agriculture_param = {'name': 'agriculture',
                    'num_classes': 4,
                    'ignore_label': 255,
                    'eval_size': [801, 801],
                    'eval_steps': 1000,
                    'eval_list': agric_eval_list,
                    'train_list': agric_train_list,
                    'data_dir': AGRIC_DATA_DIR}
    
    camvid_param = {'name': 'camvid',
                    'num_classes': 11,
                    'ignore_label': 255,
                    'eval_size': [720, 960],
                    'eval_steps': 233,
                    'eval_list': camvid_eval_list,
                    'train_list': camvid_train_list,
                    'data_dir': CAMVID_DATA_DIR}

    model_paths = {'train': './model/cityscapes/icnet_cityscapes_train_30k.npy', 
              'trainval': './model/cityscapes/icnet_cityscapes_trainval_90k.npy',
              'train_bn': './model/cityscapes/icnet_cityscapes_train_30k_bnnomerge.npy',
              'trainval_bn': './model/cityscapes/icnet_cityscapes_trainval_90k_bnnomerge.npy',
              'others': './model/ade20k/model.ckpt-27150',
              'my_model': './snapshots/model.ckpt-60000'}
    
    ## If you want to train on your own dataset, try to set these parameters.
    others_param = {'name': 'YOUR_OWN_DATASET',
                    'num_classes': 0,
                    'ignore_label': 0,
                    'eval_size': [0, 0],
                    'eval_steps': 0,
                    'eval_list': '/PATH/TO/YOUR_EVAL_LIST',
                    'train_list': '/PATH/TO/YOUR_TRAIN_LIST',
                    'data_dir': '/PATH/TO/YOUR_DATA_DIR'}

    ## You can modify following lines to train different training configurations.
    INFER_SIZE = [800, 800, 3] 
    TRAINING_SIZE = [800, 800] 
    TRAINING_STEPS = 60001
    
    N_WORKERS = 8
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    MOMENTUM = 0.9
    POWER = 0.9
    RANDOM_SEED = 1234
    WEIGHT_DECAY = 0.0001
    SNAPSHOT_DIR = './snapshots/'
    SAVE_NUM_IMAGES = 4
    SAVE_PRED_EVERY = 500
    
    # Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
    LAMBDA1 = 0.16
    LAMBDA2 = 0.4
    LAMBDA3 = 1.0
    
    def __init__(self, dataset, is_training=False, filter_scale=1, random_scale=False, random_mirror=False):
        print('Setup configurations...')
        
        if dataset == 'ade20k':
            self.param = self.ADE20k_param
        elif dataset == 'cityscapes':
            self.param = self.cityscapes_param
        elif dataset == 'cityscapes_motion':
            self.param = self.cityscapes_motion_param
        elif dataset == 'freetech':
            self.param = self.freetech_param
        elif dataset == 'freetech_lane':
            self.param = self.freetech_lane_param
        elif dataset == 'agric':
            self.param = self.agriculture_param
        elif dataset == 'camvid':
            self.param = self.camvid_param
        elif dataset == 'others':
            self.param = self.others_param

        self.dataset = dataset
        self.random_scale = random_scale
        self.random_mirror = random_mirror
        self.is_training = is_training
        self.filter_scale = filter_scale
        
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)) and not isinstance(getattr(self, a), dict):
                print("{:30} {}".format(a, getattr(self, a)))

            if a == ("param"):
                print(a)
                for k, v in getattr(self, a).items():
                    print("   {:27} {}".format(k, v))

        print("\n")