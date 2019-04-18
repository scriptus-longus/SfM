import sfm
import numpy as np
import cv2

# read images and create obj sfm
im1 = cv2.imread('viff.003.ppm')
im2 = cv2.imread('viff.001.ppm')

# height width
H,W = im1.shape[:2]
#focal length
F = float(2360)

# camera matrix
K = np.array([[F, 0, W/2], [0,F,H/2], [0,0,1]])
sfm = sfm.SfM(W,H,K)

def mod3d(sfm, img1, img2):
  #find and match features (sift)
  pts1, pts2 = sfm.procFeaturesSIFT(img1, img2)
  
  #normalize coordinates 
  pts1n, pts2n = sfm.norm(pts1, pts2)
  #compute E with normalized points
  E = sfm.EssentialCV2(pts1n, pts2n)
  
  #recover P2 from E (P1 = I0)
  P1, P2 = sfm.findPfromE(E, pts1n, pts2n)
 
  #trinagulate points
  points3d = sfm.triang(pts1n, pts2n, P1, P2)
  #draw points
  sfm.drawPointsPLT(points3d)


mod3d(sfm, im1, im2)

