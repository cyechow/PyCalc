# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 13:57:22 2018

@author: edith.chow
"""

import numpy as np
import numpy.random as npr
import pylab
import imageio
import skimage.color
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import cv2
import argparse
import imutils
import math
import Person

#%% Define algos:
def resample(weights):
  n = len(weights)
  indices = []
  C = [0.] + [sum(weights[:i+1]) for i in range(n)]
  u0, j = npr.random(), 0
  for u in [(u0+i)/n for i in range(n)]:
    while u > C[j]:
      j+=1
    indices.append(j-1)
  return indices


def particlefilter(sequence, pos, stepsize, n):
  seq = iter(sequence)
  x = np.ones((n, 2), int) * pos                   # Initial position
  f0 = next(seq)[tuple(pos)] * np.ones(n)          # Target colour model
  yield pos, x, np.ones(n)/n                       # Return expected position, particles and weights
  for im in seq:
    np.add(x, npr.uniform(-stepsize, stepsize, x.shape), out=x, casting="unsafe")  # Particle motion model: uniform step
    x  = x.clip(np.zeros(2), np.array(im.shape)-1).astype(int) # Clip out-of-bounds particles
    f  = im[tuple(x.T)]                         # Measure particle colours
    w  = 1./(1. + (f0-f)**2)                    # Weight~ inverse quadratic colour distance
    w /= sum(w)                                 # Normalize w
    yield sum(x.T*w, axis=1), x, w              # Return expected position, particles and weights
    if 1./sum(w**2) < n/2.:                     # If particle cloud degenerate:
      x  = x[resample(w),:]                     # Resample particles according to weights
      

#%% Import video:
# download from: http://www.ee.cuhk.edu.hk/~xgwang/grandcentral.html
# filename = "grandcentral.avi"


filename = "away_from_camera.mp4"
#filename = "towards_camera.mp4"
#vid = imageio.get_reader(filename)
#nFrames = vid.get_length();

vid_data = []
vid_background_data = []

cap = cv2.VideoCapture(filename)
fgbg = cv2.createBackgroundSubtractorMOG2(history = 250, detectShadows = False)

nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--images", required=True, help="path to images directory")
# args = vars(ap.parse_args())
 
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

#%% Get the first 6000 frames
i = 0;
while(i < nFrames):
    ret, frame = cap.read() #read a frame
    
    fgmask = fgbg.apply(frame) #Use the substractor
    
    try:        
        vid_data.append(frame)
        vid_background_data.append(fgmask)
    except:
        #if there are no more frames to show...
        print('EOF')
        break
    i = i + 1
    #Abort and exit with 'Q' or ESC
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
#%%
cap.release() #release video file
cv2.destroyAllWindows() #close all openCV windows

#cv2.imshow('Frame', vid_data[0])
#cv2.imshow('Background Substraction', vid_background_data[0])

#%% Clean image
#img = vid_background_data[1000];
#ret,thresh1 = cv2.threshold( img, 200, 255, cv2.THRESH_BINARY)

#kernelOp = np.ones((2,2), np.uint8)
#kernelCl = np.ones((5,5), np.uint8)

#mask = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernelOp)
#mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)

#img_mod = mask

#pylab.imshow( vid_data[0] )
#pylab.imshow( img_mod )

#%% Find contours
#img_original = vid_data[1000]
#_, contours0, hierarchy = cv2.findContours( img_mod, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
#for cnt in contours0:
#    cv2.drawContours(img_original, cnt, -1, (0,255,0), 3, 8)
    
#%% Track people
fps = 23 # Frames per second
kernelOp = np.ones((2,2), np.uint8)
kernelCl = np.ones((8,8), np.uint8)

#Variables
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
bPersonOnScreen = []
max_p_age = fps*60 # Maximum number of frames for person to not move
pid = 1
areaTH = 200

vid_data_tracked = []
vid_data_filtered = []
vid_data_contoured = []

# Loop through images and perform pedestrian detection:
for count, imgElement in enumerate(vid_background_data, 1):
    if count < len(vid_data):
    #if count % math.floor( fps ) == 0:
        imgCopy = imgElement.copy()
        imgCopy_Original = vid_data[count].copy()
        
        # Reduce size to (1) reduce detection time and (2) improve detection accuracy:
        imgCopy = imutils.resize(imgCopy, width=min(400, imgCopy.shape[1]))
        imgCopy_Original = imutils.resize(imgCopy_Original, width=min(400, imgCopy_Original.shape[1]))
        
        # Analyze roughly every half a second
        ret,imgBin = cv2.threshold( imgCopy, 200, 255, cv2.THRESH_BINARY)
        # Open
        mask = cv2.morphologyEx(imgBin, cv2.MORPH_OPEN, kernelOp)
        # Close
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
        
        vid_data_filtered.append( mask )
        
        # Contours
        _, contours0, hierarchy = cv2.findContours( mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
        
        bPersonOnScreen = np.full(len(persons), False)
        
        for cnt in contours0:
            cv2.drawContours(imgCopy_Original, cnt, -1, (0,255,0), 3, 8)
            area = cv2.contourArea(cnt)
            if area > areaTH:
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                x,y,w,h = cv2.boundingRect(cnt)
                
                # Try to find this person in database and update the coords:
                new = True
                nPerson = 0
                for i in persons:
                    # Check if this person is proximity of the area of the person in the database
                    # Assumption is that the person is same if there is proximity
                    if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h:
                        new = False
                        i.updateCoords(cx,cy)
                        bPersonOnScreen[nPerson] = True
                        break
                    nPerson = nPerson + 1
                
                # If it's a new person, add to db with the coords:
                if new == True:
                    bPersonOnScreen = np.append(bPersonOnScreen, np.full(1, True))
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1
                
                cv2.circle(imgCopy_Original,(cx,cy), 5, (0,0,255), -1)
                imgCopy_Original = cv2.rectangle(imgCopy_Original,(x,y),(x+w,y+h),(0,255,0),2)            
                cv2.drawContours(imgCopy_Original, cnt, -1, (0,255,0), 3)
        
        vid_data_contoured.append( imgCopy_Original )
        
        imgCopyTracked = imgCopy_Original
        # Trajectories
        nPerson = 0
        nTracked = 0
        for i in persons:
            if bPersonOnScreen[nPerson] and len(i.getTracks()) >= 2:
                nTracked = nTracked + 1
                pts = np.array(i.getTracks(), np.int32)
                pts = pts.reshape((-1,1,2))
                imgCopyTracked = cv2.polylines(imgCopyTracked,[pts],False,i.getRGB())
                cv2.putText(imgCopyTracked, str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv2.LINE_AA)
            nPerson = nPerson + 1

        print(nTracked)
        vid_data_tracked.append( imgCopyTracked )
        
        
#%% Track people
fps = 23 # Frames per second

#Variables
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
bPersonOnScreen = []
max_p_age = fps*60 # Maximum number of frames for person to not move
pid = 1
areaTH = 1000

vid_data_tracked = []
vid_data_filtered = []
vid_data_contoured = []

# Loop through images and perform pedestrian detection:
for count, imgElement in enumerate(vid_data, 1):
    # Analyze roughly every half a second
    if count % math.floor( fps ) == 0:
        imgCopy = imgElement.copy()
        imgCopy_Original = imgElement.copy()
        
        # Reduce size to (1) reduce detection time and (2) improve detection accuracy:
        imgCopy = imutils.resize(imgCopy, width=min(400, imgCopy.shape[1]))
        imgCopy_Original = imutils.resize(imgCopy_Original, width=min(400, imgCopy_Original.shape[1]))
        
        # detect people in the image
        (rects, weights) = hog.detectMultiScale(imgCopy_Original, winStride=(4, 4),
             padding=(8, 8), scale=1.05)
     
        # draw the original bounding boxes
        for (x, y, w, h) in rects:
            cv2.rectangle(imgCopy, (x, y), (x + w, y + h), (0, 0, 255), 2)
     
        # apply non-maxima suppression to the bounding boxes using a
        # fairly large overlap threshold to try to maintain overlapping
        # boxes that are still people
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
     
        # Get number of persons detected previous:
        bPersonOnScreen = np.full(len(persons), False)
        
        # draw the final bounding boxes
        for (xA, yA, xB, yB) in pick:
            area = xB * yB
            if area > areaTH:
                cv2.rectangle(imgCopy_Original, (xA, yA), (xB, yB), (0, 255, 0), 2)
                
                # Try to find this person in database and update the coords:
                new = True
                nPerson = 0
                for i in persons:
                    # Check if this person is proximity of the area of the person in the database
                    # Assumption is that the person is same if there is proximity
                    if abs(xA-i.getX()) <= xB and abs(yA-i.getY()) <= yB:
                        new = False
                        i.updateCoords(xA,yA)
                        bPersonOnScreen[nPerson] = True
                        break
                    nPerson += 1
                
                # If it's a new person, add to db with the coords:
                if new == True:
                    bPersonOnScreen = np.append(bPersonOnScreen, np.full(1, True))
                    p = Person.MyPerson(pid, xA, yA, max_p_age)
                    persons.append(p)
                    pid += 1
        
        vid_data_contoured.append(imgCopy_Original)
        
        imgCopyTracked = imgCopy_Original.copy()
        # Trajectories
        nPerson = 0
        nTracked = 0
        for i in persons:
            #if bPersonOnScreen[nPerson] and len(i.getTracks()) >= 2:
            if len(i.getTracks()) >= 2:
                nTracked += 1
              
                pts = np.array(i.getTracks(), np.int32)
                pts = pts.reshape((-1,1,2))
                imgCopyTracked = cv2.polylines(imgCopyTracked,[pts],False,i.getRGB())
                cv2.putText(imgCopyTracked, str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv2.LINE_AA)
            nPerson += 1

        vid_data_tracked.append( imgCopyTracked )

#%% Detect Face Function
def detect_faces(b_cascade, f_cascade, fp_cascade, colored_img, scaleFactor = 1.1, minNeighbors=5, minSizeFaces=(0,0), minSizeBodies=(0,0)):
 #just making a copy of image passed, so that passed image is not changed 
 img_copy = colored_img.copy()          
 
 #convert the test image to gray image as opencv face detector expects gray images
 gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)          
 
 #let's detect multiscale (some images may be closer to camera than others) images
 faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSizeFaces, maxSize=minSizeBodies);
 faces_profile = fp_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSizeFaces, maxSize=minSizeBodies);
 bodies = b_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSizeBodies);
 
 #go over list of faces and draw them as rectangles on original colored img
 for (x, y, w, h) in faces:
      cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
      
 for (x, y, w, h) in faces_profile:
      cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 255, 255), 2)

 for (x, y, w, h) in bodies:
      cv2.rectangle(img_copy, (x, y), (x+w, y+h), (255, 0, 0), 2)
 
 return img_copy

#%% Grayscale to RGB
def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#%% Video Data Analysis Variables
fps = 23 # Frames per second

#Variables
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
bPersonOnScreen = []
max_p_age = fps*60 # Maximum number of frames for person to not move
pid = 1
areaTH = 1000

vid_data_tracked = []
vid_data_filtered = []
vid_data_contoured = [] 

#%% Haar Cascade Detection Tracking
#load cascade classifier training file for haarcascade 
pathTrainingData = "data/haarcascade_fullbody.xml"
haar_cascade = cv2.CascadeClassifier(pathTrainingData)

pathTrainingData = "data/haarcascade_frontalface_default.xml"
haar_face_cascade = cv2.CascadeClassifier(pathTrainingData)

pathTrainingData = "data/haarcascade_profileface.xml"
haar_profile_face_cascade = cv2.CascadeClassifier(pathTrainingData)

#%% Track
vid_data_faces_haar = []
# Loop through images and perform pedestrian detection:
for count, imgElement in enumerate(vid_data, 1):
    # Analyze roughly every half a second
    if count % math.floor( fps ) == 0:
        imgCopy = imgElement.copy()
        imgCopy = detect_faces(haar_cascade, haar_face_cascade, haar_profile_face_cascade, imgCopy, scaleFactor=1.01, minNeighbors=2, minSizeBodies=(40,70))
        vid_data_faces_haar.append(convertToRGB(imgCopy))
    count += 1

#%%
vid_data_tracked = []
# Loop through images and perform pedestrian detection:
for count, imgElement in enumerate(vid_data, 1):
    # Analyze roughly every half a second
    if count % math.floor( fps ) == 0:
        imgCopy = imgElement.copy()
        #convert the test image to gray image as opencv face detector expects gray images 
        gray = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2GRAY)
        
        faces = haar_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=2, minSize=(40,70));
        
        # Get number of persons detected previous:
        bPersonOnScreen = np.full(len(persons), False)
        
        #go over list of faces and draw them as rectangles on original colored 
        for (x, y, w, h) in faces:
            cv2.rectangle(imgCopy, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Try to find this person in database and update the coords:
            new = True
            nPerson = 0
            for i in persons:
                # Check if this person is proximity of the area of the person in the database
                # Assumption is that the person is same if there is proximity
                if abs(xA-i.getX()) <= xB and abs(yA-i.getY()) <= yB:
                    new = False
                    i.updateCoords(xA,yA)
                    bPersonOnScreen[nPerson] = True
                    break
                nPerson += 1
            
            # If it's a new person, add to db with the coords:
            if new == True:
                bPersonOnScreen = np.append(bPersonOnScreen, np.full(1, True))
                p = Person.MyPerson(pid, xA, yA, max_p_age)
                persons.append(p)
                pid += 1
        
        imgCopyTracked = imgCopy.copy()
        # Trajectories
        nPerson = 0
        nTracked = 0
        for i in persons:
            #if bPersonOnScreen[nPerson] and len(i.getTracks()) >= 2:
            if len(i.getTracks()) >= 2:
                nTracked += 1
                pts = np.array(i.getTracks(), np.int32)
                pts = pts.reshape((-1,1,2))
                imgCopyTracked = cv2.polylines(imgCopyTracked,[pts],False,i.getRGB())
                cv2.putText(imgCopyTracked, str(i.getId()), (i.getX(),i.getY()), font, 0.3, i.getRGB(), 1, cv2.LINE_AA)
            nPerson += 1

        vid_data_tracked.append( imgCopyTracked )

#%% LBP Face Detection Tracking
#load cascade classifier training file for LBP 
pathTrainingData = "data/lbpcascade_profileface.xml"
lbp_cascade = cv2.CascadeClassifier(pathTrainingData)

#%% Export to GIF
#imageio.mimsave('away_from_cam_v3.gif', vid_data_tracked, duration = 0.5)
#imageio.mimsave('towards_cam_v1.gif', vid_data_tracked, duration = 23/60 )
#imageio.mimsave('towards_cam_haar_v1.gif', vid_data_faces_haar, duration = 20/60 )

#imageio.mimsave('away_from_cam_haar_v1.gif', vid_data_faces_haar, duration = 20/60 )

writer = imageio.get_writer('away_from_cam_haar_v1.mp4',fps=fps)
for im in vid_data_faces_haar:
    writer.append(im[:,:,1])
writer.close()

#%%
imageio.mimsave('towards_cam_haar_v2.gif', vid_data_tracked, duration = 20/60 )

#%% Extract frames to array
vid_data = [];
for i in range(0, 20):
    img = vid.get_data(i)
    vid_data.append( skimage.color.rgb2gray(img) )

# 'size': (720, 480)
x0 = np.array([240, 360])

#%% Run filter:
for im, p in zip(vid_data, particlefilter(vid_data, x0, 8, 100)): # Track the square through the sequence
    pos, xs, ws = p
    position_overlay = np.zeros_like(im)
    position_overlay[np.array(pos).astype(int)] = 1
    particle_overlay = np.zeros_like(im)
    particle_overlay[tuple(xs.T)] = 1
    draw()
    time.sleep(0.3)
    clf()                                           # Causes flickering, but without the spy plots aren't overwritten
    imshow(im,cmap=cm.gray)                         # Plot the image
    spy(position_overlay, marker='.', color='b')    # Plot the expected position
    spy(particle_overlay, marker=',', color='r')    # Plot the particles
    display.clear_output(wait=True)
    display.display(show())