# SfM
This is a small sfm module written in python. 

## Requirements:
  - python3  (tested: python 3.6.9)
  - opencv v.3 (tested 4.4.1)
  - tested on ubuntu 18.04
  
## how it works:
  1. find and match features (SHI-TOM, ORB etc..)
    1.1 normalize coordinates
  2. calculate Essentil Matrix from normalized coordinates
  3. extract Cameras (assume P1 = [diag(1,1,1) | 0])
  5. linear triangulation

## how to use:
  ```
  python3 sfm.py
  ```
  or
  ```
  chmod +x sfm.py
  ./sfm.py
  ```
 
## TODO:
  - automatic calibration (find focal length)
  - n-views
  - surfaces
  - extracting camera position
 
## resources:
  - Multiple view geometry in computer Vision (Richard Hartley)
  
  
