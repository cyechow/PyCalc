# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:57:22 2018

@author: edith.chow
"""

import numpy as np
import numpy.random as npr
from pylab import *
import time
import imageio
import skimage.color
from imutils.object_detection import non_max_suppression
from imutils import paths
import cv2
import imutils
import math
import Person
from matplotlib import pyplot as plt
from matplotlib import path
import os

#%% Import video:
filename = "MLSE_KISS_GP080002.mp4"
vid_reader = imageio.get_reader(filename)
nFrames = vid_reader.get_length();

fgbg = cv2.createBackgroundSubtractorMOG2(history = 250, detectShadows = False)

bgPath = 'MLSE_Background_Gate6.png'
bgImg = imageio.imread(bgPath)

#%% Get the first N frames
nStop = 300
vid_data = []
vid_background_data = []
for i, frame in enumerate(vid_reader):
    if i == nStop:
        break
    
    h, w, l = frame.shape
    #frame = cv2.resize( frame, (h, h) )

    fgmask = fgbg.apply(frame) #Use the substractor
    
    try:        
        vid_data.append(frame)
        vid_background_data.append(fgmask)
    except:
        #if there are no more frames to show...
        print('EOF')
        break

vid_reader.close()

#%% TF Imports & Variables
import tensorflow as tf
from paths import MODEL_ZOO_DIR, MODELS_OBJ_DETECTION_DIR, VIDEOS_DIR_IN, VIDEOS_DIR_OUT
from object_detection.utils import label_map_util

bbox_width_threshold = 0.25
bbox_height_threshold = 1
confidence_threshold = 0.50

# 241 ms, 37 COCO mAP[^1]
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_lowproposals_coco_2018_01_28'

# 771 ms, 36 COCO mAP[^1]
MODEL_NAME = 'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = os.path.join(MODEL_ZOO_DIR, MODEL_NAME, 'frozen_inference_graph.pb')
# List of the strings that is used to add the correct label for each box.
PATH_TO_LABELS = os.path.join(MODELS_OBJ_DETECTION_DIR, 'mscoco_label_map_all.pbtxt')
# Maximum number of (consecutive) label indices to include.
NUM_CLASSES = 90

#%% Load TF Model
print("Loading frozen TensorFlow model into memory...")
start_time = time.time()
# Load a (frozen) TensorFlow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
print("Finished in {}".format(time.time() - start_time))

# Label maps' map indices to category names, so that when our convolution network predicts 5, we
# know that this corresponds to airplane.
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map=label_map,
    max_num_classes=NUM_CLASSES,
    use_display_name=True
)
category_index = label_map_util.create_category_index(categories)

#%% Template Matching
def match_person( template, img, bDraw ):
    result = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 );
    
    _minVal, _maxVal, minLoc, maxLoc = cv2.minMaxLoc(result, None)
    
    # TM_CCOEFF_NORMED match is the max value (i.e. brightest in the result):
    # TM_SQDIFF_NORMED match is the min value (i.e. darkest in the result):
    matchLoc = minLoc
    
    #print(maxLoc)
    #print(minLoc)
    if bDraw:
        cv2.rectangle(img, matchLoc, (matchLoc[0] + template.shape[1], matchLoc[1] + template.shape[0]), (255,255,0), 2, 8, 0 )
        cv2.rectangle(result, matchLoc, (matchLoc[0] + template.shape[1], matchLoc[1] + template.shape[0]), (0,0,0), 2, 8, 0 )
    
        #plt.imshow(template)
        #plt.title('Template - Person')
        
        plt.figure(figsize=(16,5))
        plt.subplot(1,2,1)
        plt.imshow(img)
        plt.title('Source')
        plt.subplot(1,2,2)
        plt.imshow(result)
        plt.title('Result')
        
    return matchLoc
    
#%% Detect Change (Motion Detection)
def detect_change( frame1, frame2, bPlot ):
    gs1 = fgbg.apply(frame1)
    gs2 = fgbg.apply(frame2)
    
    dI = cv2.absdiff(gs1, gs2)
    #cv2.adaptiveThreshold(dI,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,12)
    dThresh = cv2.threshold(dI, 0, 255, cv2.THRESH_BINARY)[1]
    
    kernelOp = np.ones((1,1), np.uint8)
    kernelCl = np.ones((12,12), np.uint8)
    dM = cv2.morphologyEx(dThresh, cv2.MORPH_OPEN, kernelOp)
    dM = cv2.morphologyEx(dM, cv2.MORPH_CLOSE, kernelCl)
    
    if bPlot:
        plt.figure(figsize=(16,5))
        plt.subplot(1,3,1)
        plt.imshow(frame1)
        plt.title('Frame 1')
        plt.subplot(1,3,2)
        plt.imshow(frame2)
        plt.title('Frame 2')
        plt.subplot(1,3,3)
        plt.imshow(dM)
        plt.title('Movement')
        
    return dM
    

#%% Display Tracked Images:
def show_images(processed_data, aFrames):
    n = len(aFrames)
    if n == 1:
        plt.figure(figsize=(16,9))
        plt.imshow(processed_data[aFrames[0]])
    elif n == 2:
        plt.figure(figsize=(16,5))
        plt.subplot(1,2,1)
        plt.imshow(processed_data[aFrames[0]])
        plt.subplot(1,2,2)
        plt.imshow(processed_data[aFrames[1]])
    else:
        r = ceil(n / 3)
        c = 3
        plt.figure(figsize=(32,9))
        nFrame = 0
        for i in np.arange(r):
            plt.subplot(r,c,nFrame + 1)
            plt.imshow(processed_data[aFrames[nFrame]])
            nFrame += 1
            if nFrame < n:
                plt.subplot(r,c,nFrame + 1)
                plt.imshow(processed_data[aFrames[nFrame]])
            nFrame += 1
            if nFrame < n:
                plt.subplot(r,c,nFrame + 1)
                plt.imshow(processed_data[aFrames[nFrame]])
            nFrame += 1

#%% Compute Intersection
def intersect( x1min, x1max, y1min, y1max, x2min, x2max, y2min, y2max ):
    a1 = (x1max - x1min)*(y1max - y1min)
    a2 = (x2max - x2min)*(y2max - y2min)
    #A = a1 + a2
    # xmin,xmax,ymin,ymax
    coords = [max(x1min, x2min), min( x1max, x2max), max(y1min, y2min), min(y1max, y2max)]
    bValid = 0
    if coords[0] < coords[1] and coords[2] < coords[3]:
        bValid = 1
    dArea = 0
    dArea1 = 0
    dArea2 = 0
    dAreaPct = 0
    if bValid:
        dArea = (coords[1] - coords[0]) * (coords[3] - coords[2])
        dArea1 = dArea/a1
        dArea2 = dArea/a2
        dAreaPct = max( dArea1, dArea2)
    
    return [bValid, dArea, dArea1, dArea2, dAreaPct, coords]

#%% Sub-Image
def getPersonImg(imgOriginal,x,y,w,h,x_res,y_res,db):
    dw = np.arange(max(0,x-db),min(x_res-1,x+w+db))
    dh = np.arange(max(0,y-db),min(y_res-1,y+h+db))
    return np.flipud(np.rot90(imgOriginal[dh[None,:],dw[:,None]]))

#%% Get Potential Persons
def detect_persons_using_mask(f, pid, img, dM, bDraw, x_res, y_res, db, x_ub, x_lb, y_ub, y_lb):
    nValidContours = 0
    areaThresh = 200 # Must be at least 200px^2 to be valid as a potential person
    areaPctThresh = 0.25
    max_p_age = fps*60 # Maximum number of frames for person to not move
    
    aX = []
    aY = []
    aW = []
    aH = []
    aPersons = []
    imgRet = img.copy()
    _, contours, hierarchy = cv2.findContours( dM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area > areaThresh:
            nValidContours = nValidContours + 1
            x,y,w,h = cv2.boundingRect(cnt)

            # Use only if within the region of interest
            if y >= y_lb and y <= y_ub and x >= x_lb and x <= x_ub:
                # Check if already detected, merge if necessary:
                nNewPersons = 0
                bMerged = False
                for p in aPersons:
                    pX = p.getX()
                    pY = p.getY()
                    pDim = p.getLastImg().shape
                    pW = pDim[1] - 2*db
                    pH = pDim[0] - 2*db
                    bValid, dArea, dArea1, dArea2, dAreaPct, coords = intersect(x, x+w, y, y+h, pX, pX + pW, pY, pY + pH)
                    #print('Merging intersect check: Coords' + str(coords) + ' Area: ' + str(dArea) + ' Pct: ' + str(dAreaPct) + ' ' + str([x, x+w, y, y+h, pX + 0.5*db, pX + pDim[1] - 0.5*db, pY + 0.5*db, pY + pDim[0] - 0.5*db]))
                    
                    xm = min(x, pX)
                    ym = min(y, pY)
                    wm = max(x+w, pX+pW) - xm
                    hm = max(y+h, pY+pH) - ym
                    if bValid is 1 and (dAreaPct > areaPctThresh or dArea > 50):
                        # Merge??
                        #print(str(nValidContours) + ' merging with ' + str(nNewPersons))
                        imgPerson = getPersonImg(img, xm, ym, wm, hm, x_res, y_res, db)
                        #p = Person.MyPerson(pid, f, xm, ym, max_p_age, imgPerson )
                        aPersons[nNewPersons].rewriteFrame( f, xm, ym, wm, hm, imgPerson )
                        bMerged = True
                                        
                        #print('[' + str(nNewPersons+1) + '/' + str(len(aPersons)) +'] Checking X: ' + str(x) + ' Y: ' + str(y) + ' [' + str(w) + ',' + str(h) + '] against X: ' + str(pX) + ' Y:' + str(pY) + ' Dim: ' + str(pDim))
                        continue
                    nNewPersons = nNewPersons + 1
                
                if bDraw:
                    cv2.drawContours(imgRet, cnt, -1, (0,255,0), 3)
                
                if not bMerged:
                    #print('[' + str(nNewPersons) + '] Adding new person: X:' + str(x) + ' Y:' + str(y) + '. W: ' + str(w) + ' H: ' + str(h))
                    aX.append(x)
                    aY.append(y)
                    aW.append(w)
                    aH.append(h)
                    imgPerson = getPersonImg(img,x,y,w,h,x_res,y_res,db)
                    p = Person.MyPerson(pid, f, x, y, w, h, max_p_age, imgPerson )
                    aPersons.append(p)
                    pid += 1
            
    if bDraw:
        draw_persons( imgRet, aPersons)
    
    return [nValidContours, aPersons, aX, aY, aW, aH, imgRet]

#%% Detect Persons w/ TF Coco
def detect_persons_using_classifier(detection_graph, img, frame_num, pid, db):
    sess = tf.Session(graph=detection_graph)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
    bbox_width_threshold = 1
    bbox_height_threshold = 1
    confidence_threshold = 0.50
    
    x_res = img.shape[1]
    y_res = img.shape[0]
    
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3].
    image_np_expanded = np.expand_dims(img, axis=0)

    # Detection.
    (detected_boxes, scores, classes, num) = sess.run(
        fetches=[detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded},
    )

    # Get coordinates of all bounding boxes of detected persons.
    data_rows = []
    aPersons = []
    bounding_boxes = []
    for i in range(len(classes[0])):
        confidence = scores[0][i]
        obj_class = category_index[classes[0][i]]['name']
        if confidence > confidence_threshold and obj_class == 'person':
            box = detected_boxes[0][i]  # [y_min, x_min, y_max, x_max] normalized
            print('{} {}: {}'.format(
                round(confidence * 100, 2),
                category_index[classes[0][i]]['name'],
                box,
            ))

            # (box[0], box[1], box[2], box[3]) = (y_min, x_min, y_max, x_max)
            x_min = int(box[1] * x_res)
            y_min = int(box[0] * y_res)
            box_width = int((box[3] * x_res) - x_min)
            box_height = int((box[2] * y_res) - y_min)
            box_coords = (x_min, y_min, box_width, box_height)

            # Ignore nonsensical box sizes.
            valid_width = box_width < (x_res * bbox_width_threshold)
            valid_height = box_height < (y_res * bbox_height_threshold)
            if valid_width and valid_height:
                bounding_boxes.append(box_coords)
                data_rows.append([
                    frame_num,
                    frame_num + x,
                    confidence,
                    box_width,
                    box_height,
                    x_min,
                    y_min,
                ])
                print(box_coords, box_width, box_height)
                
                iArea = get_area_index(round(x_min + 0.5*box_width),round(y_min + 0.5*box_height))
                imgPerson = getPersonImg(img, x_min, y_min, box_width, box_height, x_res, y_res, db)
                p = Person.MyPerson(pid, frame_num, x_min, y_min, iArea, box_width, box_height, 0, imgPerson )
                aPersons.append(p)
                pid += 1
    
    img_classified = img.copy()
    for box in bounding_boxes:
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(img_classified, p1, p2, (200, 0, 0), 3)
        
    plt.imshow(img_classified)
    
    return aPersons

#%% Draw Persons
def draw_persons(img, aPersons, bShowDebugLogs):
    for p in aPersons:
        x = p.getX()
        y = p.getY()
        pDim = p.getLastImg().shape
        w = pDim[1] - 2*db
        h = pDim[0] - 2*db
        
        cx = x + int(0.5*w)
        cy = y + int(0.5*h)
        
        cv2.circle(img,(cx,cy), 5, (0,0,255), -1)
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        
        if bShowDebugLogs:
            print('Drawing ' + str(p.getId()) + ' Geometry: ' + str([x,y,w,h]) + ' Centroid: ' + str([cx, cy]))
        #cv2.drawContours(imgRet, cnt, -1, (0,255,0), 3)
        
    return img
        
#%% Confirm Persons
# Returns existing list with new frames added, if necessary, and list of new persons to be added
def confirm_detection(aPersons, aTest, f, widthThresh, heightThresh, x_res, y_res, db, bShowDebugLogs):
    aNewPersons = []
    nNewPerson = 0
    for pn in aTest:
        x = pn.getX()
        y = pn.getY()
        pDim = pn.getLastImg().shape
        w = pDim[1] - 2*db
        h = pDim[0] - 2*db
        
        bNew = True
        nPerson = 0
        for p in aPersons:
            # Check if any overlap:
            pX = p.getX()
            pY = p.getY()
            pDim = p.getLastImg().shape
            pW = pDim[1] - 2*db
            pH = pDim[0] - 2*db
            bValid, dArea, dArea1, dArea2, dAreaPct, coords = intersect(x, x+w, y, y+h, pX, pX + pW, pY, pY + pH)
            
            if bValid is 1 and dAreaPct > areaPctThresh:
                #print(str(nValidContours) + ' merging with ' + str(nNewPersons))
                # There is an overlap, merge the two:
                xm = min(x, pX)
                ym = min(y, pY)
                wm = max(x+w, pX+pW) - xm
                hm = max(y+h, pY+pH) - ym

                if wm < widthThresh and hm < heightThresh:
                    if bShowDebugLogs:
                        print('[' + str(pn.getId()) + '] Contour overlapping with, merging into, (' + str(p.getId()) + '). Area: ' + str(dArea) + ' 1: ' + str(dArea1) + ' 2: ' + str(dArea2) + ' Pct: ' + str(dAreaPct) + '. New values: ' + str([xm, ym, wm, hm]) + ' Inputs:' + str([x, x+w, y, y+h, pX, pX + pW, pY, pY + pH]))

                    imgPerson = getPersonImg(imgOriginal, xm, ym, wm, hm, x_res, y_res, db)
                    f_last = p.getLastFrame()
                    if f == f_last:
                        p.rewriteFrame(f,xm,ym,wm,hm,imgPerson)
                    else:
                        p.addFrame(f,xm,ym,wm,hm,imgPerson)
                    aPersons[nPerson] = p
                    bNew = False

            nPerson += 1
        
        if bNew:
            if bShowDebugLogs:
                print('[' + str(pn.getId()) + '] Adding new person: ' + str([x, y, w, h]))
            aNewPersons.append(pn)
        nNewPerson += 1
    
    return [aPersons, aNewPersons]

#%% Merge Persons
def merge_detected(vid_data, A, B, bUseOverlap, x_res, y_res, db, bShowDebugLogs):
    f_first = min(A.getFrameAtIdx(0), B.getFrameAtIdx(0))
    f_last = max(A.getLastFrame(), B.getLastFrame())
    
    if bShowDebugLogs:
        print('Merging... f1: ' + str(f_first) + ' f2: ' + str(f_last) + '. ID_A: ' + str(A.getId()) + '. ID_B: ' + str(B.getId()))
        print('A: ' + str(A.getFrames()) + '. B: ' + str(B.getFrames()))
    
    C = Person.MyPerson(A.getId(), f_first, 0, 0, 0, 0, 0, A.getImg(0))
    fA = A.getFrames()
    fB = B.getFrames()
    fA_min = min(fA)
    fA_max = max(fA)
    fB_min = min(fB)
    fB_max = max(fB)
    for f_curr in np.arange(f_first, f_last + 1):
        if f_curr >= fA_min and f_curr <= fA_max:
            try:
                idx_A = A.getIndexOfFrame(f_curr)
            except ValueError:
                print('Looking for ' + str(f_curr) + ' in ' + str(A.getId()) + ': ' + str(A.getFrames()))
        else:
            idx_A = 0
            
        if f_curr >= fB_min and f_curr <= fB_max:
            idx_B = B.getIndexOfFrame(f_curr)
        else:
            idx_B = 0
        
        if bShowDebugLogs:
            print('Frame ' + str(f_curr) + ' idx_A: ' + str(idx_A) + ' idx_B: ' + str(idx_B) + ' -- ' + str([fA_min, fA_max, fB_min, fB_max]))
        
        if f_curr < fA_min:
            if f_curr == f_first:
                if bShowDebugLogs:
                    print('Rewriting frame using B: ' + str(f_curr))
                C.rewriteFrame(f_curr, B.getXAtIndex(idx_B), B.getYAtIndex(idx_B), B.getWidth(idx_B), B.getHeight(idx_B), B.getImg(idx_B))
            else:
                if bShowDebugLogs:
                    print('Adding frame using B: ' + str(f_curr))
                C.addFrame(f_curr, B.getXAtIndex(idx_B), B.getYAtIndex(idx_B), B.getWidth(idx_B), B.getHeight(idx_B), B.getImg(idx_B))
        elif f_curr < fB_min:
            if f_curr == f_first:
                if bShowDebugLogs:
                    print('Rewriting frame using A: ' + str(f_curr))
                C.rewriteFrame(f_curr, A.getXAtIndex(idx_A), A.getYAtIndex(idx_A), A.getWidth(idx_A), A.getHeight(idx_A), A.getImg(idx_A))
            else:
                if bShowDebugLogs:
                    print('Adding frame using A: ' + str(f_curr))
                C.addFrame(f_curr, A.getXAtIndex(idx_A), A.getYAtIndex(idx_A), A.getWidth(idx_A), A.getHeight(idx_A), A.getImg(idx_A))
        else:
            if bShowDebugLogs:
                print('Frame ' + str(f_curr) + ' merging frames from A and B.')
                
            xA = A.getXAtIndex(idx_A)
            yA = A.getYAtIndex(idx_A)
            wA = A.getWidth(idx_A)
            hA = A.getHeight(idx_A)
            xB = B.getXAtIndex(idx_B)
            yB = B.getYAtIndex(idx_B)
            wB = B.getWidth(idx_B)
            hB = B.getHeight(idx_B)
            
            xm = min(xA, xB)
            ym = min(yA, yB)
            wm = max(xA + wA, xB + wB) - xm
            hm = max(yA + hA, yB + hB) - ym
            
            if bUseOverlap:
                # xmin,xmax,ymin,ymax
                bValid, dArea, dArea1, dArea2, dAreaPct, overlap_coords = intersect(xA, xA+wA, yA, yA+hA, xB, xB + wB, yB, yB + hB)
                xm = overlap_coords[0]
                ym = overlap_coords[2]
                wm = overlap_coords[1] - xm
                hm = overlap_coords[3] - ym

            if bShowDebugLogs:
                print('[' + str(A.getId()) + '] Merging frame ' + str(f_curr) + ' with this person (' + str(B.getId()) + '). New values: ' + str([xm, ym, wm, hm]) + '. Using overlap: ' + str(bUseOverlap))
            
            imgPerson = getPersonImg(vid_data[f_curr].copy(), xm, ym, wm, hm, x_res, y_res, db)
            if f_curr == f_first:
                C.rewriteFrame(f_curr, xm, ym, wm, hm, imgPerson)
            else:
                C.addFrame(f_curr, xm, ym, wm, hm, imgPerson)
        
        if bShowDebugLogs:
            idx_C = C.getIndexOfFrame(f_curr)
            xc = C.getXAtIndex(idx_C)
            yc = C.getYAtIndex(idx_C)
            wc = C.getWidth(idx_C)
            hc = C.getHeight(idx_C)
            print('Checking merge for frame ' + str(f_curr) + ': ' + str([xc, yc, wc, hc]))
            
    return C

#%% Clean Detected Persons List
# Removes any overlapping persons and merges history
def clean_detected(vid_data, aPersons, areaPctThresh, areaThresh, widthThres, heightThresh, x_res, y_res, db, bShowDebugLogs):
    nPerson = 0
    bRemoved = np.full(len(aDetectedPersons), False, dtype=bool)
    for p in aPersons:
        # Check if current is already taken care of and merged with a different person:
        if bRemoved[nPerson] == False:
            x = p.getX()
            y = p.getY()
            w = p.getLastWidth()
            h = p.getLastHeight()
            if bShowDebugLogs:
                print('Checking ' + str(p.getId()) + '. ' + str([x, y, w, h]))
            for i in np.arange(len(aPersons)):
                if i == nPerson:
                    continue
                else:
                    p_check = aPersons[i]
                    pX = p_check.getX()
                    pY = p_check.getY()
                    pW = p_check.getLastWidth()
                    pH = p_check.getLastHeight()
                    bValid, dArea, dArea1, dArea2, dAreaPct, coords = intersect(x, x+w, y, y+h, pX, pX + pW, pY, pY + pH)
                    
                    #if p.getId() == 10:
                    #    print('Against ' + str(p_check.getId()) + ' Area: ' + str(dArea) + ' Pct: ' + str(dAreaPct) +  ' Inputs: ' + str([x, x+w, y, y+h, pX, pX + pW, pY, pY + pH]))
                
                    if bValid is 1 and (dAreaPct > areaPctThresh or dArea > areaThresh):
                        xm = min(x, pX)
                        ym = min(y, pY)
                        wm = max(x+w, pX+pW) - xm
                        hm = max(y+h, pY+pH) - ym
                        
                        if dAreaPct > 0.7 or ( dAreaPct <= 0.7 and dAreaPct > areaPctThresh and wm < widthThresh and hm < heightThresh ):
                            # If need to merge, update current and mark this as to be removed and merged:
                            if bShowDebugLogs:
                                print('[' + str(p.getId()) + '] Merging with this person (' + str(p_check.getId()) + ')')
                            p_merged = merge_detected(vid_data, p, p_check, dAreaPct > 0.8, x_res, y_res, db, bShowDebugLogs)
                            aPersons[nPerson] = p_merged
                            bRemoved[i] = True
                        #else:
                            # Separate the 
                            
                    # TODO: Check template matching as well?
                    
        nPerson += 1
    
    aCleaned = []
    for i in np.arange(len(aPersons)):
        if bRemoved[i] == False:
            aCleaned.append(aPersons[i])
    if bShowDebugLogs:
        print('Existing persons: ' + str(len(aPersons)) + ' Confirmed persons: ' + str(len(aCleaned)))
    return aCleaned

#%% Fill-in-the-blanks
def fill_blanks(vid_data, aPersons, x_res, y_res, db, bShowDebugLogs):
    nPerson = 0
    for p in aPersons:
        f = p.getFrames()
        if len(f) > 1:
            f = np.sort(f)
            df = np.diff(f)
            f_missing = df[df > 1]
            nMissing = len(f_missing)
            if nMissing > 0:
                if bShowDebugLogs:
                    print('Missing ' + str(nMissing) + ' frames.')
                for f_curr in np.arange(f[0], f[len(f)-1]):
                    if ( f_curr in f ) == False:
                        if bShowDebugLogs:
                            print('Adding ' + str(f_curr) + ' into ID ' + str(p.getId()))
                        f_prev = f_curr - 1
                        idx_prev = p.getIndexOfFrame(f_prev)
                        x = p.getXAtIndex(idx_prev)
                        y = p.getYAtIndex(idx_prev)
                        w = p.getWidth(idx_prev)
                        h = p.getHeight(idx_prev)
                        imgPerson = getPersonImg(vid_data[f_curr].copy(), x, y, w, h, x_res, y_res, db)
                        p.addFrame(f_curr, x, y, w, h, imgPerson)
                
                aPersons[nPerson] = p
        nPerson += 1
    return aPersons

#%% Security Analyses
def get_area_index(cx,cy):
    # 0: Out of scope
    # 1: Pre-screen Area
    x_tl_psa = 1167
    y_tl_psa = 434
    x_bl_psa = 1159
    y_bl_psa = 707
    x_tr_psa = 1774
    y_tr_psa = 542
    x_br_psa = 1784
    y_br_psa = 786
    psa_path = path.Path([(x_bl_psa,y_bl_psa),(x_tl_psa,y_tl_psa),(x_tr_psa,y_tr_psa),(x_br_psa,y_br_psa)])
    
    # 2: Pre-detector pass
    # Divest time + bag search time
    #(time)
    x_tl_bag = 998
    y_tl_bag = 315
    x_bl_bag = 961
    y_bl_bag = 684
    x_tr_bag = 1125
    y_tr_bag = 345
    x_br_bag = 1085
    y_br_bag = 700
    bag_path = path.Path([(x_bl_bag,y_bl_bag),(x_tl_bag,y_tl_bag),(x_tr_bag,y_tr_bag),(x_br_bag,y_br_bag)])
    
    #Percentage/type of prohibited items
    #(count)
    #Duration of discussion
    #(time)
    
    # 3: Metal detector
#    x1_md
#    y1_md
#    x2_md
#    y2_md
#    x3_md
#    y3_md
#    x4_md
#    y4_md
    #Primary screening time (magnetometer and table item check)
    #(time)
    #Percentage of alarms at the WTMD
    #(count)
    
    # 4: Post-detector Area
    #Secondary screening/wanding time
    #(time)
    #Revest time (collecting items)
    #(time)
#    x1_pda
#    y1_pda
#    x2_pda
#    y2_pda
#    x3_pda
#    y3_pda
#    x4_pda
#    y4_pda
    
    if psa_path.contains_point([cx,cy]):
        return 1
    elif bag_path.contains_point([cx,cy]):
        return 2
    else:
        return 0
    

#%% [MAIN] Track people using morphology
fps = 30 # Frames per second
x_res = 1920
y_res = 1080
db = 5 #px

kernelOp = np.ones((2,2), np.uint8)
kernelCl = np.ones((12,12), np.uint8)

#Variables
font = cv2.FONT_HERSHEY_SIMPLEX
max_p_age = fps*60 # Maximum number of frames for person to not move
pid = 1
areaThresh = 50
areaPctThresh = 0.25
widthThresh = 200
heightThresh = 200
pid = 1

# Set region of interest, defined by 4 corners:
y_ub = 1000
y_lb = 300
x_ub = 1920
x_lb = 0

# Outputs
aDetectedPersons = []
vid_data_tracked = []
vid_data_filtered = []
vid_data_contoured = []

bShowDebugLogs = False

t_start = time.time()

# Number of frames to process:
nProcessedFrames = 300 #len(vid_data)

#%%
# Process the first second
vid_identification = []

persons_per_frame = []
for i in np.arange(2):
    frame_num = i + 1
    
    # Movement between 3 frames:
    img1 = vid_data[frame_num]
    img2 = vid_data[frame_num + 2]
    
    # get greyscale:
    gs_Img1 = fgbg.apply(img1)
    gs_Img2 = fgbg.apply(img2)
    
    dI = cv2.absdiff(gs_Img1, gs_Img2)
    dThresh = cv2.threshold(dI, 25, 255, cv2.THRESH_BINARY)[1] #cv2.adaptiveThreshold(dI,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,12)
    #dPx = cv2.sumElems(dThresh)[0]
    
    kernelOp = np.ones((1,1), np.uint8)
    kernelCl = np.ones((8,8), np.uint8)
    dM = cv2.morphologyEx(dThresh, cv2.MORPH_OPEN, kernelOp)
    dM = cv2.morphologyEx(dM, cv2.MORPH_CLOSE, kernelCl)
    
#    plt.figure(figsize=(16,9))
#    plt.imshow(dM)
#    plt.title('Thresh Adjusted')
#    
#    plt.figure(figsize=(16,9))
#    plt.imshow(vid_data[frame_num].copy())
#    plt.title('Original')
    
    imgOriginal = vid_data[frame_num].copy()
    imgOutput = vid_data[frame_num].copy()
    _, contours0, hierarchy = cv2.findContours( dM, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    
    imgs = []
    detected_boxes = []
    count = 0
    pid = 0
    
    for cnt in contours0:
        area = cv2.contourArea(cnt)
        #print('Contour area: {}'.format(area))
        if area > areaThresh:
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x,y,w,h = cv2.boundingRect(cnt)
    
            # Draw contour if within the region of interest
            if y >= y_lb and y <= y_ub and x >= x_lb and x <= x_ub:
                #print('[' + str(count) + '] Potential person area: ' + str(area) + ' ' + str([x,y,w,h]) + ' Center: ' + str([cx,cy]))
             
                detected_boxes.append([x,y,w,h])
                imgPerson = getPersonImg(imgOriginal,x,y,w,h,x_res,y_res,db)
                
                imgs.append(imgPerson)
                
                cv2.drawContours(imgOutput, cnt, -1, (0,255,0), 3, 8)
                
        count = count + 1


    # Merging pass before adding to detected persons for this frame 
    print('Total detected: ' + str(len(detected_boxes)))
    bRemoved = np.full(len(detected_boxes), False, dtype=bool)    
    frame_persons = []
    
    # x,y,w,h
    for nBox in np.arange(len(detected_boxes)):
        box = detected_boxes[nBox]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        cv2.rectangle(imgOutput,(x,y),(x+w,y+h),(255,0,0),8)
        if bRemoved[nBox]:
            continue
        
        # Check if the box is small, try to find the person within this region:
        widthThresh = 40
        heightThresh = 50
        
        for n in np.arange(len(detected_boxes)):
            test_box = detected_boxes[n]
            if n == nBox or bRemoved[n]:
                continue
            
            bValid, dArea, dArea1, dArea2, dAreaPct, coords = intersect(x, x + w, y, y + h, test_box[0], test_box[0] + test_box[2], test_box[1], test_box[1] + test_box[3])
            
            #if bValid:
            #    print('[' + str(frame_num) + '] Intersecting: ' + str(box) + ' and ' + str(test_box))
            
            # coords: xmin,xmax,ymin,ymax
            xi1 = coords[0]
            xi2 = coords[1]
            yi1 = coords[2]
            yi2 = coords[3]
            
            wi = xi2 - xi1
            hi = yi2 - yi1
            
            dBoxArea = box[2]*box[3]
            dTestArea = test_box[2]* test_box[3]
            
            # Merge together if:
            #   - Is overlapping and over 0.25 coverage
            #   - Merged dimensions are valid
            if dAreaPct > 0.25 and wi < 150 and hi < 200:
                bRemoved[n] = True
                
                xm = min(box[0], test_box[0])
                ym = min(box[1], test_box[1])
                wm = max(box[0] + box[2], test_box[0] + test_box[2]) - xm
                hm = max(box[1] + box[3], test_box[1] + test_box[3]) - ym
                
                print('Curr: ' + str(box) + ' Test: ' + str(test_box) + ' Coords: ' + str(coords) + ' Old: ' + str(detected_boxes[nBox]) + ' New: ' + str([xm,ym,wm,hm]))
                detected_boxes[nBox] = [xm,ym,wm,hm]
            # Check if area is too small, if it is, check proximity to current:
            elif w <= widthThresh and h <= heightThresh:
                proximityThresh = 10
                x1 = x - proximityThresh
                x2 = x + w + proximityThresh
                y1 = y - proximityThresh
                y2 = y + h + proximityThresh
                
                proximity_results = intersect(x1, x2, y1, y2, test_box[0], test_box[0] + test_box[2], test_box[1], test_box[1] + test_box[3])
                
                if proximity_results[0] == 1:
                    bRemoved[n] = True
                    print('Box too small, merging with nearby')
                    xm = min(x, test_box[0])
                    ym = min(y, test_box[1])
                    wm = max(x + w, test_box[0] + test_box[2]) - xm
                    hm = max(y + h, test_box[1] + test_box[3]) - ym
                    print('Curr: ' + str(box) + ' Test: ' + str(test_box) + ' Coords: ' + str(coords) + ' Old: ' + str(detected_boxes[nBox]) + ' New: ' + str([xm,ym,wm,hm]))
                    detected_boxes[nBox] = [xm,ym,wm,hm]
            
            # If it does intersect midly, split it:
            elif bValid == 1:
                bRemoved[n] = True
                
                x1 = box[0]
                x2 = box[0] + box[2]
                y1 = box[1]
                y2 = box[1] + box[3]
                
                if dArea1 > dArea2:
                    x1 = test_box[0]
                    x2 = test_box[0] + test_box[2]
                    y1 = test_box[1]
                    y2 = test_box[1] + test_box[3]
                    
                # Where are x1 and x2 moving?
                x1_test = x1
                x2_test = x2
                if xi1 <= x1 and xi2 > x1 and xi2 <= x2:
                    print('Moving xmin from ' + str(x1) + ' to ' + str(xi2))
                    x1_test = xi2
                    x2_test = x2
                elif xi1 > x1 and xi2 > x2:
                    print('Moving xmax from ' + str(x2) + ' to ' + str(xi1))
                    x1_test = x1
                    x2_test = xi1
                
                # Where are y1 and y2 moving?
                y1_test = y1
                y2_test = y2
                if yi1 <= y1 and yi2 > y1 and yi2 <= y2:
                    print('Moving ymin from ' + str(y1) + ' to ' + str(yi2))
                    y1_test = yi2
                    y2_test = y2
                elif yi1 > y1 and yi2 > y2:
                    print('Moving ymax from ' + str(y2) + ' to ' + str(yi1))
                    y1_test = y1
                    y2_test = yi1
                
                if dArea1 > dArea2:
                    print('Curr: ' + str(box) + ' Test: ' + str(test_box) + ' Coords: ' + str(coords) + ' Old: ' + str(detected_boxes[n]) + ' New: ' + str([x1_test, y1_test, x2_test - x1_test, y2_test - y1_test]))
                    detected_boxes[n] = [x1_test, y1_test, x2_test - x1_test, y2_test - y1_test]
                else:
                    print('Curr: ' + str(box) + ' Test: ' + str(test_box) + ' Coords: ' + str(coords) + ' Old: ' + str(detected_boxes[nBox]) + ' New: ' + str([x1_test, y1_test, x2_test - x1_test, y2_test - y1_test]))
                    detected_boxes[nBox] = [x1_test, y1_test, x2_test - x1_test, y2_test - y1_test]
            
        nBox += 1

    # Lastly, add to final persons list for this frame
    nPSA = 0
    nBag = 0
    nOther = 0
    for i in np.arange(len(detected_boxes)):
        if bRemoved[i]:
            continue
        box = detected_boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        imgPerson = getPersonImg(imgOriginal,x,y,w,h,x_res,y_res,db)
        
        detect_persons_using_classifier(detection_graph, imgPerson, frame_num, pid, x_res, y_res, db)
        
        iArea = get_area_index(round(x+0.5*w), round(y+0.5*h))
        if iArea == 0:
            nOther += 1
        elif iArea == 1:
            nPSA += 1
        elif iArea ==2:
            nBag += 1
        p = Person.MyPerson(pid, frame_num, x, y, iArea, w, h, 0, imgPerson )
        frame_persons.append(p)
        
        # Draw bounding boxes onto image
        cv2.rectangle(imgOutput,(x,y),(x+w,y+h),(0,255,0),2)
    
    print('[' + str(frame_num) + '] Total: ' + str(len(frame_persons)) + ' Pre-Screening Area: ' + str(nPSA) + ' Bag-Check: ' + str(nBag) + ' Other: ' + str(nOther))
    
    persons_per_frame.append(frame_persons)
    vid_identification.append(imgOutput)
#    plt.figure(figsize=(16,9))
#    plt.imshow(imgOriginal)
#    plt.title('Contours Added')

if False:
    writer = imageio.get_writer('MLSE_KISS_GP080002_CVMovement_1sec_v2.mp4',fps=fps)
    for im in vid_identification:
    #for im in vid_data_contoured:
        writer.append_data(im[:,:,1])
    writer.close()

#%%
# Loop through images and perform pedestrian detection:
for count, imgElement in enumerate(vid_background_data, 1):
    frame_num = count - 1
    if frame_num < nProcessedFrames:
        if True: #bShowDebugLogs:
            print('Frame ' + str(frame_num))
            
        # Show time estimation every 30 frames
        if frame_num and frame_num % 30 == 0:
            t_current = time.time()
            time_so_far = t_current - t_start
            time_per_frame = time_so_far / frame_num
            estimated_time_remaining = (nProcessedFrames - frame_num) * time_per_frame
            print('%i of %i: estimated time remaining = %s seconds'
                  % (frame_num, nProcessedFrames, round(estimated_time_remaining)))
            
        imgCopy = imgElement.copy()
        imgCopy_Original = vid_data[frame_num].copy()
        
        imgOriginal = vid_data[frame_num].copy()
        
        # Find persons already detected in new image
        #print('[' + str(frame_num) + '] Persons detected: ' + str(len(aDetectedPersons)))
        if len(aDetectedPersons) > 0:
            if bShowDebugLogs:
                print('Updating already detected persons (adding new frame)...')
            nPerson = 0
            for p in aDetectedPersons:
                imgPerson = p.getLastImg()
                pLoc = match_person( imgPerson, imgOriginal, False )
                
                # Check if the location makes sense:
                w = p.getLastWidth()
                h = p.getLastHeight()
                
                dx = pLoc[0] - p.getX()
                dy = pLoc[1] - p.getY()
                
                if abs(dx) <= w and abs(dy) <= h:
                    if bShowDebugLogs:
                        print('[' + str(frame_num) + '] Updating detected person ' + str(p.getId()) + '. Location: ' + str(pLoc) + ' dX,dY: ' + str([dx,dy]))
                    
                    # Get image, give it some buffer:
                    imgPerson = getPersonImg(imgOriginal, pLoc[0], pLoc[1], w, h, x_res, y_res, db)
                    p.addFrame(frame_num, pLoc[0], pLoc[1], w, h, imgPerson)
                    aDetectedPersons[nPerson] = p
                nPerson = nPerson + 1

            # Clean detected:
            aDetectedPersons = fill_blanks(vid_data, aDetectedPersons, x_res, y_res, db, bShowDebugLogs)
            aDetectedPersons = clean_detected(vid_data, aDetectedPersons, areaPctThresh, areaThresh, widthThresh, heightThresh, x_res, y_res, db, bShowDebugLogs)
        
        # Detect Movements
        aPersonsMoving = []
        nPersonsMoving = 0
        if frame_num > 1:
            if bShowDebugLogs:
                print('Detecting movement...')
            imgMovement = vid_data[frame_num].copy()
            dM = detect_change( vid_data[frame_num - 1].copy(), imgMovement, False)
            [nPersonsMoving, aPersonsMoving, aXMoving, aYMoving, aWMoving, aHMoving, imgMovement] = detect_persons_using_mask( frame_num, pid, imgMovement, dM, False, x_res, y_res, db, x_ub, x_lb, y_ub, y_lb )
            pid += len(aPersonsMoving)
            [aDetectedPersons, newPersons] = confirm_detection(aDetectedPersons, aPersonsMoving, frame_num, widthThresh, heightThresh, x_res, y_res, db, bShowDebugLogs)
            aDetectedPersons.extend(newPersons)
            
            # Clean detected:
            #aDetectedPersons = fill_blanks(vid_data, aDetectedPersons, x_res, y_res, db, bShowDebugLogs)
            aDetectedPersons = clean_detected(vid_data, aDetectedPersons, areaPctThresh, areaThresh, widthThresh, heightThresh, x_res, y_res, db, bShowDebugLogs)
        
        # Detect People (Binary Image)
        # Only do it in the beginning to start off the process
        if frame_num < 2:
            if bShowDebugLogs:
                print('Detecting potential new persons using binary imaging...')
            # Get binary image by thresholding
            ret,imgBin = cv2.threshold( imgCopy, 0, 255, cv2.THRESH_BINARY)
            # Open
            mask = cv2.morphologyEx(imgBin, cv2.MORPH_OPEN, kernelOp)
            # Close
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
            
            vid_data_filtered.append( mask )
            
            # Binary Image Contour Mask
            [nPersonsContour, aPersonsContour, aXContour, aYContour, aWContour, aHContour, imgCopy_Original] = detect_persons_using_mask( frame_num, pid, imgCopy_Original, mask, False, x_res, y_res, db, x_ub, x_lb, y_ub, y_lb )
            pid += len(aPersonsContour)
            # Try to find this person in database and update the coords, otherwise add new person:
            [aDetectedPersons, newPersons] = confirm_detection(aDetectedPersons, aPersonsContour, frame_num, widthThresh, heightThresh, x_res, y_res, db, bShowDebugLogs)
            
            # Add new persons to persons array:
            aDetectedPersons.extend(newPersons)
            
            # Clean detected:
            #aDetectedPersons = fill_blanks(vid_data, aDetectedPersons, x_res, y_res, db, bShowDebugLogs)
            aDetectedPersons = clean_detected(vid_data, aDetectedPersons, areaPctThresh, areaThresh, widthThresh, heightThresh, x_res, y_res, db, bShowDebugLogs)
        
        # Draw rectangles:
        imgCopy_Original = draw_persons(imgCopy_Original, aDetectedPersons, bShowDebugLogs)
        
        vid_data_contoured.append( imgCopy_Original )
        
        imgCopyTracked = imgCopy_Original.copy()
        
        # Trajectories
        nPerson = 0
        nTracked = 0
        for p in aDetectedPersons:
            pFrame = p.getLastFrame()
            if pFrame == frame_num and len(p.getTracks()) >= 2:
                nTracked = nTracked + 1
                pts = np.array(p.getTracks(), np.int32)
                pts = pts.reshape((-1,1,2))
                imgCopyTracked = cv2.polylines(imgCopyTracked,[pts],False,p.getRGB())
                cv2.putText(imgCopyTracked, str(p.getId()),(p.getX(),p.getY()),font,0.3,p.getRGB(),1,cv2.LINE_AA)
            nPerson = nPerson + 1

        if bShowDebugLogs:
            print('Tracking ' + str(nTracked) + ' persons.')
        vid_data_tracked.append( imgCopyTracked )
        
#%% Save Video/GIF
nVer = 9
vid_data_output = []

for frame_num in np.arange(len(vid_data_tracked)):
    f = frame_num + 1
    n = 0
    img = vid_data[frame_num].copy()
    for p in aDetectedPersons:
        if p.checkFrameExist(f):
            idx = p.getIndexOfFrame(f)
            x = p.getXAtIndex(idx)
            y = p.getYAtIndex(idx)
            w = p.getWidth(idx)
            h = p.getHeight(idx)
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            print( str(f) + '-' + str(n) + ' [' + str(p.getId()) + '] Coords: ' + str([x,y,w,h]) )
        n += 1
    vid_data_output.append(img)

plt.figure(figsize=(16,9))
plt.imshow(img)

writer = imageio.get_writer('MLSE_KISS_GP080002_CVMorph_v' + str(nVer) + '.mp4',fps=fps)
for im in vid_data_output:
#for im in vid_data_contoured:
    writer.append_data(im[:,:,1])
writer.close()

imageio.mimsave('MLSE_KISS_GP080002_CVMorph_v' + str(nVer) + '.gif', vid_data_output, duration = 1/fps )
