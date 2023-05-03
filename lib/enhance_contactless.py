#!/usr/bin/env python
from __future__ import print_function
import sys
PY3 = sys.version_info[0] == 3
import os
from lib.FingerprintImageEnhancer import FingerprintImageEnhancer
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

def enhance_fingerprint(img):
    open_cv_image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.flip(open_cv_image, 1)
    img = img[max(img.shape[0] - int(img.shape[1]*1.6), 0):, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    img = (255-img)
    img = cv2.medianBlur(img, 5)
    img = cv2.medianBlur(img, 3)
    img = cv2.medianBlur(img, 3)

    image_enhancer = FingerprintImageEnhancer() 
    enh_img, ridge = image_enhancer.enhance(img) 

    enh_img = 255*enh_img
    ridge = 255 * ridge.astype(np.uint8)

    enh_img = coherence_filter(ridge, sigma = 11, str_sigma = 53, blend = 1, iter_n = 5, blockSize=3)
    
    return enh_img

def main(image):
    enh_img = enhance_fingerprint(image)
    return image, enh_img