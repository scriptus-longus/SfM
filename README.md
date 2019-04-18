# SfM
small sfm module for python3

## Requirements:
  - python3  (tested: python 3.5.2)
  - opencv v.3 (tested 3.4.2)
  - tested on ubuntu 16.04
  
## how it works:
  1. find and match features using SIFT or SHI-TOMASI
    1.1 normalize coordinates
  2. calculate Essentil Matrix from normalized coordinates
  3. find camera P2 (camera parameters of Second cam)
  5. linear triangulation

## how to use:
  ```
  python3 example.py
  ```
  [!Screenshot](images/test.png)
  
## resorces:
  - https://github.com/alyssaq/3Dreconstruction/ (example code)
  - https://github.com/geohot/twitchslam (example code)
  - https://cmsc426.github.io/sfm/
  - https://hub.packtpub.com/exploring-structure-motion-using-opencv/
  - http://www.robots.ox.ac.uk/~vgg/data/data-mview.html
  
  
