# SfM
small sfm module for python3

# Requirements:
  - python3  (tested: python 3.5.2)
  - opencv v.3 (tested 3.4.2)
  - tested on ubuntu 16.04
  
# how it works:
  1. find and match features using SIFT or SHI-TOMASI
    1.1 normalize coordinates
  2. calculate Essential Matrix from normalized coordinates
  3. find camera P2 (camera parameters of Second cam)
  5. linear triangulation
  
