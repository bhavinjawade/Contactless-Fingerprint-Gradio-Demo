import numpy as np
import cv2
import os
import json

def segment_distals(img):
    scale = 0.05
    inv = 1/scale

    scale_save = 0.5
    scale_save_inv = 1 / scale_save
    img_save = cv2.resize(img, (0,0), fx=scale_save, fy=scale_save) 

    x,y,w,h = int(img.shape[1]*scale)-120,20,120,150 # x, y, w, h
    
    small = cv2.resize(img, (0,0), fx=scale, fy=scale) 

    mask = np.zeros(small.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    
    mask, bgdModel, fgdModel = cv2.grabCut(small, mask, (x,y,w,h), bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    cv2.rectangle(small, (x, y), (x+w, y+h), (0,0,255), 2)
    
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    cv2.imwrite("./small_mask.jpg", mask2 )

    mask2 = cv2.resize(mask2, (img_save.shape[1],img_save.shape[0])) 

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    opening = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=7)

    img_seg = img_save*opening[:,:,np.newaxis]
    cv2.imwrite("./small_mask.jpg", mask2)

    cv2.imwrite("./segmented.jpg", img_seg)
    cv2.imwrite("./orgimg.jpg", small)
    cv2.imwrite("./mask.jpg",opening * 255)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    src = img_seg
    src = cv2.GaussianBlur(src, (3, 3), 0)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    # cv2.imwrite("./gradient.jpg", grad)
    edges = cv2.Canny(grad,100,100)
    # cv2.imwrite("./edges.jpg", edges)


    ret, thresh = cv2.threshold(opening*255,50,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    top = []
    for i in range(len(contours)):
        hull = cv2.convexHull(contours[i],returnPoints=False)
        defects = cv2.convexityDefects(contours[i], hull)
        #cv2.drawContours(img, [hull], -1, (255, 0, 0), 2)
        for j in range(defects.shape[0]):
            s,e,f,d = defects[j,0]
            start = tuple(contours[i][s][0])
            end = tuple(contours[i][e][0])
            far = tuple(contours[i][f][0])
            
            top.append([d, start,end, far])

    top.sort(key=lambda x: x[0], reverse=True)
    convex_points = []
    for d,start,end,far in top[:3]:
        cv2.putText(img_seg,str(d), far, cv2.FONT_HERSHEY_SIMPLEX, 5, 255)
        cv2.line(img_seg,start,end,[0,255,0],2)
        cv2.circle(img_seg,far,20,[0,0,255],-1)
        convex_points.append(far)

    convex_points.sort(key=lambda x: x[1])

    x1, y1 = 0, [i for i,x in enumerate(opening[:,-1]) if x == 1][0]
    x2, y2 = convex_points[0]
    x3, y3 = convex_points[1]
    x4, y4 = convex_points[2]
    x5, y5 = 0, [i for i,x in enumerate(opening[:,-1]) if x == 1][-1]
    boxes = []
    width_finger = max((y3-y2),(y4-y3))

    topleft = (x2 - 100, y2 - width_finger)
    bottomright = (x2 - 100 + int(width_finger * 1.5), y2)
    cv2.rectangle(img_seg, topleft, bottomright, (0,255,255), 5)
    boxes.append([int(topleft[0]), int(topleft[1]), int(bottomright[0]), int(bottomright[1])])

    topleft = (x3 - int(width_finger/2), y2)
    bottomright = (x3 - int(width_finger/2) + 200 + int(width_finger * 1.5), y3)
    cv2.rectangle(img_seg, topleft, bottomright, (0,255,255), 5)
    boxes.append([int(topleft[0]), int(topleft[1]), int(bottomright[0]), int(bottomright[1])])

    topleft = (x3 - 150, y3)
    bottomright = (x3 - 150 + int(width_finger * 1.5), y4)
    cv2.rectangle(img_seg, topleft, bottomright, (0,255,255), 5)
    boxes.append([int(topleft[0]), int(topleft[1]), int(bottomright[0]), int(bottomright[1])])

    topleft = (x4 - 100, y4)
    bottomright = (x4 - 100 + int(width_finger * 1.5), y4 + width_finger)
    cv2.rectangle(img_seg, topleft, bottomright, (0,255,255), 5)
    boxes.append([int(topleft[0]), int(topleft[1]), int(bottomright[0]), int(bottomright[1])])

    labels = [0,0,0,0]
    return img_seg

def main(file):
    img_seg = segment_distals(file)
    return img_seg

# segment_distals("2_google_41879_1_RIGHT_image_fingerprint5K9SUL2L.png")