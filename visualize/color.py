from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import numpy as np
import os
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import cv2
##############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])
def labelcolormap(N):
    '''
    returns a numpy array representing a mapping: cmap[class][0] = r, cmap[class][1] = g, cmap[class][2] = b
    '''
    if N == 35: # cityscape
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)],
                     dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b
    return cmap
class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])
    def __call__(self, gray_image):
        '''
        gray_image: 1 x H x W (expected to be tensor with channel dim before w and h)
        '''
        # import pdb; pdb.set_trace()
        shape = gray_image.shape
        color_image = torch.ByteTensor(3, shape[1], shape[2]).fill_(0)
        for label in range(len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]
        return color_image
class ColorizeNumpy(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])
    def __call__(self, gray_image):
        '''
        gray_image: H x W
        '''
        # import pdb; pdb.set_trace()
        shape = gray_image.shape
        color_image = np.zeros(shape=[shape[0], shape[1], 3], dtype=np.uint8)
        for label in range(len(self.cmap)):
            mask = (label == gray_image)
            color_image[mask, 0] = self.cmap[label][0]
            color_image[mask, 1] = self.cmap[label][1]
            color_image[mask, 2] = self.cmap[label][2]
        return color_image
color = ColorizeNumpy(19)
def colorize(gray_img):
    return color(gray_img)
CITYSCAPES_CLASSES = {
    0: [0, 0, 0],  # None
    1: [70, 70, 70],  # Buildings
    2: [190, 153, 153],  # Fences
    3: [72, 0, 90],  # Other
    4: [220, 20, 60],  # Pedestrians
    5: [153, 153, 153],  # Poles
    6: [157, 234, 50],  # RoadLines
    7: [128, 64, 128],  # Roads
    8: [244, 35, 232],  # Sidewalks
    9: [107, 142, 35],  # Vegetation
    10: [0, 0, 255],  # Vehicles
    11: [102, 102, 156],  # Walls
    12: [220, 220, 0]  # TrafficSigns
}
def as_cityscapes_palette(frame):
    """ Transforms the frame to the Carla cityscapes pallete.
    Note: this conversion is slow.
    """
    result = np.zeros((frame.shape[0], frame.shape[1], 3),
                      dtype=np.uint8)
    for key, value in CITYSCAPES_CLASSES.items():
        result[np.where(frame == key)] = value
    return result