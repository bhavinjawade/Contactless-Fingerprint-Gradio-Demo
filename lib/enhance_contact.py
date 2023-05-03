#!/usr/bin/env python
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3
import os
from lib.enhance_ak import FingerprintImageEnhancer
from skimage.transform import resize


if PY3:
    xrange = range

import numpy as np
import cv2

def coherence_filter(img, sigma = 11, str_sigma = 11, blend = 0.5, iter_n = 4, blockSize=29):
    h, w = img.shape[:2]

    for i in xrange(iter_n):
        gray = img # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        eigen = cv2.cornerEigenValsAndVecs(gray, str_sigma, blockSize)
        eigen = eigen.reshape(h, w, 3, 2)  # [[e1, e2], v1, v2]
        x, y = eigen[:,:,1,0], eigen[:,:,1,1]

        gxx = cv2.Sobel(gray, cv2.CV_32F, 2, 0, ksize=sigma)
        gxy = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=sigma)
        gyy = cv2.Sobel(gray, cv2.CV_32F, 0, 2, ksize=sigma)
        gvv = x*x*gxx + 2*x*y*gxy + y*y*gyy
        m = gvv < 0

        ero = cv2.erode(img, None)
        dil = cv2.dilate(img, None)
        img1 = ero
        img1[m] = dil[m]
        img = np.uint8(img*(1.0 - blend) + img1*blend)
    return img


def main(image):

    # img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_enhancer = FingerprintImageEnhancer() 
    enh_img = image_enhancer.enhance(img) 
    enh_img = 255*enh_img
    return image, enh_img