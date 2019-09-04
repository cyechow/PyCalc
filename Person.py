# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 16:43:05 2018

@author: edith.chow
From http://www.femb.com.mx/people-counter/people-counter-8-finding-movement/
"""

from random import randint
import numpy as np

class MyPerson:
    tracks = []
    def __init__(self, i, f, xi, yi, iArea, hi, wi, max_age, img):
        self.i = i
        self.f = [f]
        self.x = [xi]
        self.y = [yi]
        self.tracks = [[xi+0.5*wi,yi+0.5*hi]]
        self.area = [iArea]
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.img = [img]
        self.w = [wi]
        self.h = [hi]
        self.done = False
        self.state = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None
    def getRGB(self):
        return (self.R,self.G,self.B)
    def getImg(self, idx):
        return self.img[idx]
    def getLastImg(self):
        return self.img[len(self.img)-1]
    def getLastWidth(self):
        return self.w[len(self.w)-1]
    def getWidth(self, idx):
        return self.w[idx]
    def getLastHeight(self):
        return self.h[len(self.h)-1]
    def getHeight(self, idx):
        return self.h[idx]
    def getTracks(self):
        idx = np.argsort( np.array(self.f))
        t = np.array(self.tracks)
        return t[idx].tolist()
    def getId(self):
        return self.i
    def getState(self):
        return self.state
    def getDir(self):
        return self.dir
    def getX(self):
        return self.x[len(self.x) - 1]
    def getXAtIndex(self, idx):
        return self.x[idx]
    def getY(self):
        return self.y[len(self.y) - 1]
    def getYAtIndex(self, idx):
        return self.y[idx]
    def getFrames(self):
        return self.f
    def getFrameAtIdx(self, idx):
        return self.f[idx]
    def getIndexOfFrame(self, f):
        return self.f.index(f)
    def checkFrameExist(self, f):
        try:
            self.f.index(f)
            return True
        except ValueError:
            return False
    def getLastFrame(self):
        return self.f[len(self.f) - 1]
    def addFrame(self, fn, xn, yn, iArea, wn, hn, img):
        self.age = self.age + 1
        self.x.append(xn)
        self.y.append(yn)
        self.w.append(wn)
        self.h.append(hn)
        self.f.append(fn)
        self.tracks.append([xn+0.5*wn,yn+0.5*hn])
        self.area.append(iArea)
        self.img.append(img)
    def rewriteFrame(self, fn, xn, yn, iArea, wn, hn, img):
        idx = self.f.index(fn)
        self.x[idx] = xn
        self.y[idx] = yn
        self.w[idx] = wn
        self.h[idx] = hn
        self.tracks[idx] = [xn+0.5*wn,yn+0.5*hn]
        self.area[idx] = iArea
        self.img[idx] = img
    def updateImg(self, img):
        self.img.append(img)
    def setDone(self):
        self.done = True
    def timedOut(self):
        return self.done
    def going_UP(self,mid_start,mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                if self.tracks[-1][1] < mid_end and self.tracks[-2][1] >= mid_end: #cruzo la linea
                    state = '1'
                    self.dir = 'up'
                    return True
            else:
                return False
        else:
            return False
    def going_DOWN(self,mid_start,mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                if self.tracks[-1][1] > mid_start and self.tracks[-2][1] <= mid_start: #cruzo la linea
                    state = '1'
                    self.dir = 'down'
                    return True
            else:
                return False
        else:
            return False
    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True
class MultiPerson:
    def __init__(self, persons, xi, yi):
        self.persons = persons
        self.x = [xi]
        self.y = [yi]
        self.tracks = []
        self.R = randint(0,255)
        self.G = randint(0,255)
        self.B = randint(0,255)
        self.done = False
        