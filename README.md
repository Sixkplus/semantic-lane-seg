# semantic-lane-seg
Real-Time Semantic Segmentation and Lane Segmentation Using 2-branches approach

- To train the pure segmentation model, type

CUDA_VISIBLE_DEVICES=0,1,2 python train_seg.py --dataset freetech --random-mirror --random-scale --filter-scale 1 --train-beta-gamma --update-mean-var 

- To train the 2-branch model with 2-paths annotations 

CUDA_VISIBLE_DEVICES=0,1,2 python train_seg_lane.py --dataset freetech_lane --random-mirror --random-scale --filter-scale 1 --train-beta-gamma --update-mean-var