#!/usr/bin/env python3.5
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

import feature as ft
import camera as cam

# insert focal length of video
F = float(2360.0)

class SfM(object):
  def __init__(self, W, H, K):
    self.K = K
    self.W = W
    self.H = H

  # conv cart coords to hom
  def cvtCart2Hom(self,pts):
    if pts.ndim ==1:
      return np.hstack([pts, 1])
    return np.asarray(np.vstack([pts, np.ones(pts.shape[1])]))

  def norm(self, pts1, pts2):
    pts1 = self.cvtCart2Hom(pts1.T)
    pts2 = self.cvtCart2Hom(pts2.T)
 
    pts1ret = np.dot(np.linalg.inv(self.K), pts1)
    pts2ret = np.dot(np.linalg.inv(self.K), pts2)
    return pts1ret, pts2ret

  # find match using shi-tomasi
  def procFeaturesSHI_TOM(self, img1, img2):
    kps1, des1 = ft.extractSHI_TOM(img1)
    kps2, des2 = ft.extractSHI_TOM(img2)

    pts1, pts2 = ft.matchBF(kps1, des1, kps2, des2)
    pts1, pts2 = ft.optMatch(pts1, pts2)
    return pts1, pts2
  # sift feature matching
  def procFeaturesSIFT(self, img1, img2):
    kp1, des1, kp2, des2 = ft.extractSIFT(img1, img2)
    pts1, pts2 = ft.matchFLANN(kp1, des1, kp2, des2)
    pts1, pts2 = ft.optMatch(pts1, pts2)
    return pts1, pts2

  # essential mat
  def EssentialCV2(self,p1, p2):
    pts1E = cv2.UMat(p1[:2].T)
    pts2E = cv2.UMat(p2[:2].T)
    PP = (float(self.W/2), float(self.H/2))
    F = float(self.K[0,0])

    E, mask = cv2.findEssentialMat(pts1E, pts2E, pp=PP, focal=F, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E
  
  # fond proj mat from E (P1 = [I0])
  def findPfromE(self, E, pts1, pts2):
    P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])

    posP2 = cam.findPose(E)

    # find correct pose
    for i, P2 in enumerate(posP2):
      a1 = cam.recSingPoint(pts1[:,0], pts2[:,0], P1, P2)

      P2hom = np.linalg.inv(np.vstack([P2, [0,0,0,1]]))
      a2 = np.dot(P2hom[:3, :4], a1)
      
      # find best P2
      if a1[2] > 0 and a2[2] > 0:
        ret = np.linalg.inv(np.vstack([posP2[i], [0,0,0,1]]))[:3, :4]

    return P1, ret

  def triang(self, pts1, pts2, P1, P2):
    n = pts1.shape[1]
    ret = np.ones((4, n))
   
    # triangulate 
    for i in range(n):
      A = cam.linTriang(pts1, pts2, P1, P2, i)
      
      U,S,V = np.linalg.svd(A)
      X = V[-1, :4]
      ret[:,i] = X/X[3]
    return ret 

  #display result plt
  def drawPointsPLT(self,pts):
    fig = plt.figure()
    fig.suptitle('SfM', fontsize=10)
    
    ax = fig.gca(projection='3d')
    ax.plot(pts[0], pts[1], pts[2], 'g.')
    # label 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # display 
    ax.view_init(elev=20, azim=90)
    plt.show()


if __name__ == '__main__':
  img1 = cv2.imread('viff.003.ppm')
  img2 = cv2.imread('viff.001.ppm')

  H,W, = img1.shape[:2]

  # scale image
  if W > 1024:
    downscale = 1024.0/W
    F *= downscale
    H = int(H*downscale)
    W = 1024

  # camera matrix
  K = np.array([[F,0,W/2],[0,F,H/2], [0,0,1]])

  #find features
  sfm = SfM(W,H,K)
  pts1, pts2 = sfm.procFeaturesSIFT(img1, img2)

  # convert coords to hom
  pts1hom = sfm.cvtCart2Hom(pts1.T)
  pts2hom = sfm.cvtCart2Hom(pts2.T)
  
  # normalize points
  pts1n = np.dot(np.linalg.inv(K), pts1hom)
  pts2n = np.dot(np.linalg.inv(K), pts2hom)

  # find essential mat
  E = sfm.EssentialCV2(pts1n, pts2n)
  
  P1, P2 = sfm.findPfromE(E, pts1n, pts2n)

  points3d = sfm.triang(pts1n, pts2n, P1, P2) 
  
  # sfm.drawPointsPLT(points3d)  
