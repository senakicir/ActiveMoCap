# -*- coding: utf-8 -*-
import os
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy
import cv2

for i in range(0,88):
    img1 = io.imread('test_sets/test_set_02_01/images/img_' + str(i)+ '.png')

    #img2 = io.imread('temp_main/2018-06-13-00-38/animation_1/superimposed_images/plot3d_' + str(i)+ '.png')
    #margin1 = 70
    #margin2 = 30

    #img1 = img1[margin2: img1.shape[0]-margin2, margin1: img1.shape[1]-margin1, :]

    #img1 = scipy.ndimage.interpolation.zoom(img1, [3,3,1])

    #pad_amount = int((img2.shape[0]-img1.shape[0])/2)
    img1_orig = np.array(img1[:,:,0:3])
    #img1 = np.vstack([np.zeros([pad_amount, img1_orig.shape[1], 3], dtype=np.uint8),img1_orig])
    #img1 = np.vstack([img1, np.zeros([pad_amount, img1.shape[1], 3], dtype=np.uint8)])
    large_image = cv2.resize(img1_orig, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    small_image = cv2.resize(img1_orig, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    #output = np.hstack([img1, img2[:,:,0:3]])

    io.imsave('test_sets/test_set_02_01_close/images/img_' + str(i)+ '.png', large_image)
    io.imsave('test_sets/test_set_02_01_far/images/img_' + str(i)+ '.png', small_image)
    print(i)