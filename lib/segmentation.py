import os
import numpy as np
import cv2
from tqdm import tqdm
import asyncio
from argparse import ArgumentParser
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)

model = init_detector("./lib/config.py", "./lib/epoch_20.pth", "cpu")

def segment(img, hand):
    global model
    colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200)]
    count = 0
    count_total = 0

    if ("RIGHT" == hand):
        x_margin_end = 0
        x_margin_start = 50
        mfinger = 50

    else:
        x_margin_start = -100
        x_margin_end = -50
        mfinger = 0

    bboxes = inference_detector(model, img)
    bboxes = sorted(bboxes, key = lambda x: x[-1])
    best_boxes = bboxes[0][:4]
    best_boxes = sorted(best_boxes, key = lambda x: x[1])
    width = int(sum([box[2] - box[0] for box in best_boxes]) / 4)
    annotated_img = img.copy()

    cropped_fingers = []

    for j, box in enumerate(best_boxes):
        if (j == 1):
            conf = box[-1]
            box = [int(b) for b in box[:-1]]
            annotated_img = cv2.rectangle(annotated_img, (max(box[0]+x_margin_start+mfinger, 0), box[1]), (min(box[0]+x_margin_start+mfinger+width, img.shape[0]-5), box[3]),  colors[j], 10)
            crop_img = img[box[1]:box[3], max(box[0]+x_margin_start+mfinger, 0):min(box[0]+x_margin_start+mfinger+width, img.shape[0]-5)]

        else:
            conf = box[-1]
            box = [int(b) for b in box[:-1]]
            annotated_img = cv2.rectangle(annotated_img, (max(box[0]+x_margin_start+mfinger, 0), box[1]), (min(box[0]+x_margin_start+mfinger+width, img.shape[0]-5), box[3]),  colors[j], 10)
            crop_img = img[box[1]:box[3], max(box[0]+x_margin_start+mfinger, 0):min(box[0]+x_margin_start+mfinger+width, img.shape[0]-5)]
        
        cropped_fingers.append(crop_img)
    return annotated_img, cropped_fingers

def main(img, hand):
    return segment(img, hand)    
