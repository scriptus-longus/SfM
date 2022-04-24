import numpy as np
import cv2
import features as ft
from camera import Camera
import matplotlib.pyplot as plt
from scipy.spatial import distance
from helpers import get_points, PointMap, get_cam_trajectory

class SfM:
  def __init__(self, cams=[]):
    self.cams = cams
    self.map = PointMap()

  def match_keypoints(self, cam1, cam2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(cam1.des, cam2.des, k=2)
  
    #good = []
    pts1 = []
    pts2 = []
    for m,n in matches:
      if m.distance < 0.70*n.distance:
        pts1.append(list(cam1.kps[m.queryIdx].pt))
        pts2.append(list(cam2.kps[m.trainIdx].pt))
        #good.append([m])

    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    return pts1, pts2

  def clean_points(self, x):
    ret = x.copy()
    indexes = []
    distance_map = distance.cdist(x, x)
    mean_dist = np.mean(distance_map)

    for i in range(len(x)):
      if abs(mean_dist - np.mean(distance_map[i])) > 5:
        ret = np.delete(ret, i, axis=0)
        indexes.append(i)
    return ret, indexes
     
 
  def two_view_sfm(self, two_cams=None):
    # detect and match features
    if len(self.cams) < 2:
      raise Exception("No images given")
      
    if two_cams != None:
      cam1 = two_cams[0]
      cam2 = two_cams[1]
    
    cam1 = self.cams[0]
    cam2 = self.cams[1]

    # match using SIFT
    pts1, pts2 = self.match_keypoints(cam1, cam2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]

    pts1n = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=cam1.K, distCoeffs=None)
    pts2n = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=cam2.K, distCoeffs=None)
    

    E = cam2.K.T @ F @ cam1.K

    _, R, t, mask = cv2.recoverPose(E, pts1n, pts2n)

    cam2.P = np.hstack((R, t))

    X = cv2.triangulatePoints(cam1.P, cam2.P, pts1n, pts2n)
    X /= X[3]

    return pts1, pts2, X[:3].T

  def run(self):
    camera_positions = [np.array([0,0,0])]
    if len(self.cams) < 2:
      raise Exception("No images given")
 
    self.cams[0].pts, self.cams[1].pts, pts3d = self.two_view_sfm()
    pts3d, indexes = self.clean_points(pts3d)
    for i in indexes:
      self.cams[0].pts = np.delete(self.cams[0].pts, i, axis=0)
      self.cams[1].pts = np.delete(self.cams[1].pts, i, axis=0)
    assoc_3d_points = [pts3d]

    camera_positions.append(self.cams[1].P[:, 3:].reshape(3) + camera_positions[-1])
    #self.cams[1].position = self.cams[0].position + self.cams[1].P[:, 3:].reshape(3)

    print(self.cams[0].pts.shape)
    print(pts3d.shape)

    self.map.add_points(self.cams[0].pts, pts3d)
    self.map.add_points(self.cams[1].pts, pts3d)
    print(self.map.all_3d_points)

    for i in range(2, len(self.cams)):
      cam1 = self.cams[i-1]
      cam2 = self.cams[i]

      pts1, pts2 = self.match_keypoints(cam1, cam2) 
  
      F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
      pts1 = pts1[mask.ravel() == 1]
      pts2 = pts2[mask.ravel() == 1]

      pts3d, mask = self.map.get_points(pts1)
      mask = np.array(mask)

      old_pts = pts2[mask==1]

      new_pts1 = pts1[mask==0]
      new_pts2 = pts2[mask==0]

      new_pts1n = cv2.undistortPoints(np.expand_dims(new_pts1, axis=1), cameraMatrix=cam1.K, distCoeffs=None)
      new_pts2n = cv2.undistortPoints(np.expand_dims(new_pts2, axis=1), cameraMatrix=cam2.K, distCoeffs=None)
 

      _, R, t, _ = cv2.solvePnPRansac(pts3d, old_pts, cam1.K, None)
      R, _ = cv2.Rodrigues(R)
 
      cam2.P =np.hstack((R, t))

      X = cv2.triangulatePoints(cam1.P, cam2.P, new_pts1n, new_pts2n)
      X /= X[3]

      X = X[:3].T

      X, indexes = self.clean_points(X)
      for i in indexes:
        pts1 = np.delete(pts1, i, axis=0)        
        pts2 = np.delete(pts2, i, axis=0)        

      cam1.pts = pts1
      cam2.pts = pts2


      assoc_3d_points.append(X)
      camera_positions.append(cam2.P[:, 3:].reshape(3) + camera_positions[-1])
      #cam2.position = cam1.position + cam2.P[:, 3:].reshape(3)

      self.map.add_points(new_pts1, X)  
      self.map.add_points(new_pts2, X)  
    

    return self.map, np.vstack(camera_positions) #np.vstack(camera_positions) #np.array(self.mem.all_3d_points), np.vstack(camera_positions)

if __name__ == "__main__":
  cams = [Camera("images/images_castle/100_7101.JPG"), 
          Camera("images/images_castle/100_7103.JPG"), 
          Camera("images/images_castle/100_7104.JPG"), 
          Camera("images/images_castle/100_7105.JPG"), 
          Camera("images/images_castle/100_7106.JPG")] 
          #Camera("images/images_castle/100_7107.JPG")] 
  

  sfm = SfM(cams=cams)
  point_map, cam_coords = sfm.run()   

  points = get_points(point_map) 
  #cam_coords = get_cam_trajectory(cams)

  fig = plt.figure()
  
  ax =fig.add_subplot(111, projection="3d")
  ax.set_xlim3d(-3, 3)
  ax.set_zlim3d(0, 5)
  ax.set_ylim3d(-3,3) 
 
  ax.scatter(points[:, 0], points[:, 1], points[:, 2], color="blue")
  ax.scatter(cam_coords[:, 0], cam_coords[:, 1], cam_coords[:, 2], color="green")
  plt.show()
