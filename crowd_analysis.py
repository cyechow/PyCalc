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
import cv2
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
filename = "grandcentral.avi"
vid = imageio.get_reader(filename)
nFrames = vid.get_length();

vid_data = []
vid_background_data = []

cap = cv2.VideoCapture(filename)
fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

nFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#%% Get the first 6000 frames
i = 0;
while(i < 6000):
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
img = vid_background_data[1000];
ret,thresh1 = cv2.threshold( img, 200, 255, cv2.THRESH_BINARY)

kernelOp = np.ones((2,2), np.uint8)
kernelCl = np.ones((5,5), np.uint8)

mask = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernelOp)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)

img_mod = mask

#pylab.imshow( vid_data[0] )
#pylab.imshow( img_mod )

#%% Find contours
img_original = vid_data[1000]
_, contours0, hierarchy = cv2.findContours( mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
for cnt in contours0:
    cv2.drawContours(img_original, cnt, -1, (0,255,0), 3, 8)
    
#%% Track people
fps = 23 # Frames per second
kernelOp = np.ones((2,2), np.uint8)
kernelCl = np.ones((10,10), np.uint8)

#Variables
font = cv2.FONT_HERSHEY_SIMPLEX
persons = []
max_p_age = fps*60 # Maximum number of frames for person to not move
pid = 1
areaTH = 100

vid_data_tracked = []
vid_data_filtered = []
vid_data_contoured = []

for count, imgElement in enumerate(vid_background_data, 1):
    if count % math.floor( fps ) == 0:
        imgCopy = imgElement
        # Analyze roughly every half a second
        ret,imgBin = cv2.threshold( imgCopy, 200, 255, cv2.THRESH_BINARY)
        # Open
        mask = cv2.morphologyEx(imgBin, cv2.MORPH_OPEN, kernelOp)
        # Close
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernelCl)
        
        vid_data_filtered.append( mask )
        
        # Contours
        _, contours0, hierarchy = cv2.findContours( mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
        for cnt in contours0:
            cv2.drawContours(imgCopy, cnt, -1, (0,255,0), 3, 8)
            area = cv2.contourArea(cnt)
            if area > areaTH:
                M = cv2.moments(cnt)
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                x,y,w,h = cv2.boundingRect(cnt)
                # Try to find this person in database and update the coords:
                new = True
                for i in persons:
                    # Check if this person is proximity of the area of the person in the database
                    # Assumption is that the person is same if there is proximity
                    if abs(x-i.getX()) <= w and abs(y-i.getY()) <= h:
                        new = False
                        i.updateCoords(cx,cy)
                        break
                # If it's a new person, add to db with the coords:
                if new == True:
                    p = Person.MyPerson(pid,cx,cy, max_p_age)
                    persons.append(p)
                    pid += 1     
                
                cv2.circle(imgCopy,(cx,cy), 5, (0,0,255), -1)
                img = cv2.rectangle(imgCopy,(x,y),(x+w,y+h),(0,255,0),2)            
                cv2.drawContours(imgCopy, cnt, -1, (0,255,0), 3)
        
        vid_data_contoured.append( imgCopy )
        
        imgCopyTracked = imgCopy
        # Trajectories
        for i in persons:
            if len(i.getTracks()) >= 2:
                pts = np.array(i.getTracks(), np.int32)
                pts = pts.reshape((-1,1,2))
                imgCopyTracked = cv2.polylines(imgCopyTracked,[pts],False,i.getRGB())
            cv2.putText(imgCopyTracked, str(i.getId()),(i.getX(),i.getY()),font,0.3,i.getRGB(),1,cv2.LINE_AA)
        
        vid_data_tracked.append( imgCopyTracked )

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