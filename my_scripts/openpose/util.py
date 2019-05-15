import numpy as np
import cv2
import torch
from torch.autograd import Variable

from time import time
import math
from .constants import *



def padRightDownCorner(img, stride, padValue):
    """
    Pads the original image such that the final size is a multiple of the stride.
    The padding is done in the right down corner with the provided value.

    :param img: Image to pad (H x W x C)
    :param stride: The new width and height should be a multiple of this value
    :param padValue: Value with which to pad.
    :return: The padded image
    """
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]

    pad = 4 * [None]
    pad[0] = 0  # up
    pad[1] = 0  # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride)  # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride)  # right

    return_array = np.full([h + pad[2], w + pad[3], c], fill_value=padValue, dtype=img.dtype)
    return_array[:h, :w, :] = img

    return return_array, pad


def preapreImage(img, scale, stride, padValue):
    """
    Prepares the image to be passed to the model. The image is scaled to the desired size and is padded so its
    final size is a multiple of the stride.

    :param img: Image to prepare (H x W x C)
    :param scale: Scale factor with which to rescale the image
    :param stride: The final image size should be a multiple of this value (height and width)
    :param padValue: Value with which to pad
    :return: The image ready to be passed to the model
    """
    imageToTest = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    imageToTest_padded, pad = padRightDownCorner(imageToTest, stride, padValue)
    imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:, :, :, np.newaxis]), (3, 2, 0, 1)) / 256 - 0.5  # Strange that they use 256 and not 255

    return Variable(torch.from_numpy(imageToTest_padded)).cuda()


def plot_joints(image_path, poses, model_type):
    tic = time()
    if 'COCO' in model_type:
        limbSeq = limbSeq_COCO
        colors = colors_COCO
        nb_limbs = 17
    if 'MPI' in model_type:
        limbSeq = limbSeq_MPI
        colors = colors_MPI
        nb_limbs = 14
    stickwidth = 4
    canvas = cv2.imread(image_path)  # B,G,R order
    for i in range(nb_limbs):
        for pose in poses:
            index = np.array([pose[elem] for elem in np.array(limbSeq[i]) - 1])
            if 0 in index[..., 2]:
                continue
            cur_canvas = canvas.copy()
            Y = index[..., 0]
            X = index[..., 1]
            mX = np.mean(X)
            mY = np.mean(Y)

            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    image_time = time() - tic
    #print 'Creating the image took {}'.format(image_time)
    #print '_' * 80
    return canvas

def plot_joints2(image_path, bones, model_type):
    tic = time()
    bone_connections = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [8,14], [8, 9], [9, 10], [11,14], [11, 12], [12, 13], [14, 1]]

    if 'COCO' in model_type:
        limbSeq = limbSeq_COCO
        colors = colors_COCO
        nb_limbs = 17
    if 'MPI' in model_type:
        limbSeq = limbSeq_MPI
        colors = colors_MPI
        nb_limbs = 14
        bone_connections = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [8,14], [8, 9], [9, 10], [11,14], [11, 12], [12, 13], [14, 1]]
    if 'VNect' in model_type:
        bone_connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],[0, 7], [7, 8], [8, 9], [9, 10], [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13]]
        nb_limbs = 17
        colors = colors_COCO

    stickwidth = 4
    canvas = cv2.imread(image_path)  # B,G,R order
    for i, connection in enumerate(bone_connections):
        cur_canvas = canvas.copy()

        Y = np.array([bones[0, connection[0]], bones[0, connection[1]]])
        X = np.array([bones[1, connection[0]], bones[1, connection[1]]])
        mX = np.mean(X)
        mY = np.mean(Y)

        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
        canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    image_time = time() - tic
   # print ('Creating the image took {}'.format(image_time))
   # print ('_' * 80)
    return canvas