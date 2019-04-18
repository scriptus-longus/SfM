import cv2
import numpy as np

# define variables
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)

def extractSIFT(img1, img2):
  # detect sift ft
  sift = cv2.xfeatures2d.SIFT_create()

  kps1, des1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
  kps2, des2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)
 
  return kps1, des1, kps2, des2

def extractSHI_TOM(img):
  orb = cv2.ORB_create()
  # find features and cvt to keypoints 
  kp = cv2.goodFeaturesToTrack(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 3000, 0.01, 8)
  kps = [cv2.KeyPoint(x=p[0][0], y=p[0][1], _size=20) for p in kp]

  kps, des = orb.compute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), kps) 
  return kps, des 


def matchBF(kps1, des1, kps2, des2):
  # match bf for shi-tomasi 
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
  matches = bf.match(des1, des2)

  matches = sorted(matches, key = lambda x:x.distance)
 
  goodMat = [] 
  for m in matches:
    if m.distance < 20.0: 
      goodMat.append(m) 
  pts1 = np.asarray([kps1[m.queryIdx].pt for m in goodMat])
  pts2 = np.asarray([kps2[m.trainIdx].pt for m in goodMat])

  return pts1, pts2  

def matchFLANN(kps1, des1, kps2, des2):
  # match using flann 
  flann = cv2.FlannBasedMatcher(index_params, search_params)

  matches = flann.knnMatch(des1, des2, k=2)
   
  # extract good matches
  goodMat = []
  for m,n in matches:
    if m.distance < 0.8*n.distance:
      goodMat.append(m) 

  pts1 = np.asarray([kps1[m.queryIdx].pt for m in goodMat])
  pts2 = np.asarray([kps2[m.trainIdx].pt for m in goodMat])
  return pts1, pts2
  
def optMatch(pts1, pts2):
  # find homography
  _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 100.0)
  mask = mask.ravel()

  pts1 = pts1[mask == 1]
  pts2 = pts2[mask == 1]
  return pts1, pts2


