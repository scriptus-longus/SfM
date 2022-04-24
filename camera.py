from __future__ import division
import numpy as np
import cv2
import features as ft
#from helpers import normalizeCoords

class Camera(object):
  def __init__(self, img):
    self.path = img
    self.img = cv2.imread(img)
    self.H, self.W = self.img.shape[:2]
    self.f = 1.2 * max(self.W, self.H)#2905.88
    self.K = np.array([[self.f, 0,self.W/2], [0, self.f, self.H/2], [0,0,1]], dtype="float32")
    self.P = np.eye(3,4)
    self.position = np.array([0,0,0])

    self.kps, self.des = self.detectFeatures()

  def detectFeatures(self, method="ORB", nMax=30000000, quality=0.01, distance=8):
    kps, des = ft.detect(self.img, method=method, nMax=nMax, quality=quality, distance=distance)
   
    self.pts = np.array([kp.pt for kp in  kps])
    return kps, des
