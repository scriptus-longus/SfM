from __future__ import division 
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import features as ft
from camera import Camera
from helpers import normalizeCoords, recSingPoint, findPose
from copy import copy
from scipy.optimize import minimize


def findPfromE(E, pts1, pts2):
    P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]])

    posP2 = findPose(E)

    # find correct pose
    for i, P2 in enumerate(posP2):
      a1 = recSingPoint(pts1[:,0], pts2[:,0], P1, P2)

      P2hom = np.linalg.inv(np.vstack([P2, [0,0,0,1]]))
      a2 = np.dot(P2hom[:3, :4], a1)
 
      # find best P2
      if a1[2] > 0 and a2[2] > 0:
        ret = posP2[i]

    return P1, ret

class SfM(object):
  def __init__(self, cams):
    self.cams = cams
    self.f = 2905.88

  def _match(self, c1, c2):
    matches = ft.match(c1.des, c2.des, method="FLANN", _sorted=False, distance=0.75)

    pts1 = np.array([c1.kps[m[0].queryIdx].pt for m in matches], dtype="float64")
    pts2 = np.array([c2.kps[m[0].trainIdx].pt for m in matches], dtype="float64")
    return pts1, pts2

  def _linTriang(self, pt1, pt2, P1, P2):
    A = np.asarray([(pt1[0]*P1[2] - P1[0]), 
                    (pt1[1]*P1[2] - P1[1]),
                    (pt2[0]*P2[2] - P2[0]), 
                    (pt2[1]*P2[2] - P2[1])])
    U,S,V = np.linalg.svd(A)
    return V[-1, :4]

  def _triang(self, pts1, pts2, P1, P2):
    ret = np.zeros((pts1.shape[0], 4))

    for i, (pt1, pt2) in enumerate(zip(pts1, pts2)):
      ret[i] = self._linTriang(pt1, pt2, P1, P2)
      ret[i] /= ret[i][3]
    return ret

  def _computeLoss(self, f):
    K = np.array([[f[0],0,self.W], [0,f[0],self.H], [0,0,1]])
    E = K.T.dot(self.F).dot(K)

    U,S,V =  np.linalg.svd(E)
    loss = 0.3* S[0]/S[1]
    print(S)
    return loss

  def _findPose(self, poses, x1, x2):
    P1 = np.eye(3,4)

    for i,p in enumerate(poses):
      pt = self._triang(x1, x2, P1, p)
      pt_P1 = np.asarray([x.dot(P1.T) for x in pt])
      pt_P2 = np.asarray([x.dot(p.T) for x in pt])


      if (np.all(pt_P1[:,2] > 0) or np.all(pt_P1[:, 2] < 0)) and (np.all(pt_P1[:,2] > 0) or np.all(pt_P1[:,2] < 0)):
        return P1, p

    print("error could not find correct Pose")
    return P1, poses[0]  
  
  def run(self):
    for i,c in enumerate(cams):
      if i > 0:
        cam1 = copy(c)
        cam2 = copy(cams[i-1])

        self.H, self.W = cam1.img.shape[:2]

        pts1, pts2 = self._match(cam1, cam2)


        pts1n, Ta = normalizeCoords(pts1)
        pts2n, Tb = normalizeCoords(pts2)

        # find Fundamental Matrix
        _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
        F, _ = cv2.findFundamentalMat(pts1n, pts2n, cv2.FM_RANSAC)

        self.F = Tb.T.dot(F).dot(Ta)

        cam1.pts = pts1[mask.ravel()==1]
        cam2.pts = pts2[mask.ravel()==1]

    
        # estimate K and E
        self.f = 3759.0     
 
        cam1.updateFocal(self.f)  
        cam2.updateFocal(self.f)

        cam1.updateCoords()
        cam2.updateCoords()

        K = cam1.K

        E = K.T.dot(self.F).dot(K)

        pts1norm = cam1.pts_norm
        pts2norm = cam2.pts_norm 

        # extract pose
        U,S,V = np.linalg.svd(E)

        W = np.array([[0,-1,0], [1,0,0], [0,0,1]])


        R1 = np.dot(np.dot(U, W), V)
        R2 = np.dot(np.dot(U, W.T), V)

        if np.linalg.det(R1) < 0:
          R1 = -1*R1
        if np.linalg.det(R2) < 0:
          R2 = -1*R2


        poses = np.array([np.hstack((R1,  U[:, 2].reshape(3,1))),
                          np.hstack((R1, -U[:, 2].reshape(3,1))),
                          np.hstack((R2,  U[:, 2].reshape(3,1))),
                          np.hstack((R2, -U[:, 2].reshape(3,1)))])

        # find coorect pose
        
        P1, P2 = self._findPose(poses, pts1norm, pts2norm)
        #findPfromE(E, pts1norm, pts2norm)

        self.X = self._triang(pts1norm, pts2norm, P1, P2)    #TODO: append  
        del cam1
        del cam2


if __name__ == "__main__":
  #cams = [Camera("images/viff.001.ppm"), Camera("images/viff.003.ppm")]
  cams = [Camera("images/test01.jpg"), Camera("images/test02.jpg")]
  #cams = [Camera("images/img03.jpg"), Camera("images/img04.jpg")]

  sfm = SfM(cams)
  sfm.run()

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  
  ax.scatter(sfm.X[:, 0], sfm.X[:, 1], sfm.X[:, 2])
  plt.show()
 
