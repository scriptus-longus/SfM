import cv2
import numpy as np


def detect(img, method="ORB", nMax=50000, quality=0.01, distance=5):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  orb = cv2.ORB_create(nfeatures=nMax, scoreType=cv2.ORB_FAST_SCORE)

  if method == "ORB":
    kps = orb.detect(gray, None)
    kps, des = orb.compute(gray, kps)
    
  elif method == "FAST":
    fast = cv2.FastFeatureDetector_create()

    kps = fast.detect(gray, None)
    kps, des = orb.compute(gray, kps)

  elif method == "SHI-TOM":
    kps = cv2.goodFeaturesToTrack(gray, nMax, quality, distance)
    kps = [cv2.KeyPoint(x=p[0][0], y=p[0][1], size=10) for p in kps]

    kps, des = orb.compute(gray, kps)

  elif method == "SIFT":
    sift = cv2.SIFT_create()

    kps, des = sift.detectAndCompute(img, None)     
  else:
    raise Exception("Invalid feature detection method")

  return kps, des

def match(des1, des2, method="BF", _sorted=False, distance=None):
  #gray = img.cvtColor(img, cv2.COLOR_BGR2GRAY)

  if method == "BF":
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)
  
    matches = bf.match(des1, des2)

    if _sorted:
      matches = sorted(matches, key=lambda x:x.distance)
  
    if distance != None and abs(distance) <= 1:
      idx = int(round(len(matches)*abs(distance))) 
      matches = matches[:idx]

  elif method == "FLANN":
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)

    search_params = dict(checks=80)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # filter matches
    if distance != None and abs(distance) <= 1:
      matches_good = []

      for i,(m,n) in enumerate(matches):
        if m.distance < distance*n.distance:
          matches_good.append([m,n])
      matches = matches_good
 
  return matches

def detectAndMatch(img1, img2, features="ORB", matching="BF", nMax=50000, _sorted=False, quality=0.01, distanceDet=5, distanceMat=None):
  kps1, des1 = detect(img1, method=features, nMax=nMax, distance=distanceDet)
  kps2, des2 = detect(img2, method=features, nMax=nMax, distance=distanceDet)

  matches = match(des1, des2, method=matching, _sorted=_sorted, distance=distanceMat)

  return (kps1, kps2), (des1, des2), matches

if __name__ == "__main__":
  img1 = cv2.imread("images/img3.jpg")
  img2 = cv2.imread("images/img4.jpg")

  kps, des, matches = detectAndMatch(img1, img2, features="SHI-TOM", matching="FLANN")
 

#matches = sorted(matches, key=lambda x:x.distance)

#img3 = cv2.drawMatches(img1, kps[0], img2, kps[1], matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
