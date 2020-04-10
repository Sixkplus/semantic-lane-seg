#coding:utf-8
import cv2
import os
import numpy as np
from PIL import Image, ImageDraw,ImageFont
from tqdm import trange

PATH_IMGS = 'video_output'

output_shape = (1928,1256)


fps = 15
#fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')


jpgNames = os.listdir(PATH_IMGS)


video_writer = cv2.VideoWriter(filename='demo_inter-frame.avi', fourcc=fourcc, fps=fps, frameSize=output_shape)

for i in trange(len(jpgNames)):
    curImgPath = os.path.join(PATH_IMGS, "frame"+str(i)+".png")
    img = cv2.imread(filename=curImgPath)
    video_writer.write(img)



