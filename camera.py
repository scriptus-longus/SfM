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
    self.f = 2905.88
    self.K = np.array([[self.f, 0,self.W/2], [0, self.f, self.H/2], [0,0,1]], dtype="float32")
    self._detectFeatures()

 
  def _detectFeatures(self, method="SHI-TOM", nMax=30000000, quality=0.01, distance=8):
    self.kps, self.des = ft.detect(self.img, method=method, nMax=nMax, quality=quality, distance=distance)
   
    self.pts = np.array([kp.pt for kp in  self.kps])
    self._homogene()
    self._normalizeCoords() 
    self._normalizePoints()

  def _normalizePoints(self):
    self.pts_norm = np.dot(np.linalg.inv(self.K), self.pts_h.T).T
    

  def _normalizeCoords(self):
    X = 0
    Y = 0

    for x,y in self.pts:
      X += x
      Y += y

    X /= self.pts.shape[0]
    Y /= self.pts.shape[0]

    ptsrec = np.zeros(self.pts.T.shape)
    ptsrec[0] = self.pts.T[0] - Y
    ptsrec[1] = self.pts.T[1] - X

    ptsrec = ptsrec.T

    dist = 0
    for i,p in enumerate(ptsrec):
      dist += p[0]**2 + p[1]**2

    dist /= (i+1)

    s = np.sqrt(2)/(dist**(1/2))


    T1 = np.dot(np.array([[s,0,0], [0,s,0], [0,0,1]]), np.array([[1, 0,-X], [0, 1, -Y], [0,0,1]]))

    ptshom = np.array([[p[0], p[1], 1] for p in self.pts.tolist()])

    ptsnorm = np.dot(T1, ptshom.T).T

    ret = np.zeros(ptsnorm.shape)

    for i,p in enumerate(ptsnorm):
      ret[i][0] = p[0]/p[2]
      ret[i][1] = p[1]/p[2]
      ret[i][2] = 1

    self.pts_n = ret[:, :2]
    self.T = T1

  def _homogene(self):
    self.pts_h = np.hstack((self.pts, np.ones((self.pts.shape[0], 1))))

  def updateFocal(self, f):
    self.f = f
    self.K = np.array([[self.f, 0,self.W/2], [0, self.f, self.H/2], [0,0,1]], dtype="float32")
    self._normalizePoints()

  def updateK(self, K):
    #self.K = np.array([[self.f, 0,self.W/2], [0, self.f, self.H/2], [0,0,1]], dtype="float32")
    self.f = self.K[0, 0]
    self._normalizePoints()

  def updateCoords(self):
    self._normalizeCoords()    
    self._homogene()
    self._normalizePoints()

  def updateFeatures(self):
    self._detectFeatures()

if __name__ == "__main__":
  cam = Camera("images/test01.jpg")
  #cam.detectFeatures()
  print(cam.pts_norm) 
