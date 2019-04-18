import numpy as np

def skewVec(V):
  ret = np.array([[0, -V[2], V[1]],
                  [V[2], 0, -V[0]],
                  [-V[1], V[0], 0]])
  return ret

# reconstruct single point for pose
def recSingPoint(x1, x2, m1, m2):
  A = np.vstack([np.dot(skewVec(x1), m1),
                 np.dot(skewVec(x2), m2)])

  u,s,v = np.linalg.svd(A)
  X = np.ravel(v[-1, :4])
  return X/X[3]

# linear triangulation
def linTriang(x1, x2, P1, P2, i):
  ret = np.asarray([
    (x1[0,i] * P1[2, :] - P1[0, :]),
    (x1[1,i] * P1[2, :] - P1[1, :]),
    (x2[0,i] * P2[2, :] - P2[0, :]),
    (x2[1,i] * P2[2, :] - P2[1, :])])
  return ret

# find all possible proj poses
def findPose(E):
  u,s,v = np.linalg.svd(E)

  if np.linalg.det(np.dot(u, v)) < 0:
    v = -v

  W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
  # poses
  ret = [np.vstack((np.dot(u, np.dot(W, v)).T, u[:,2])).T,
           np.vstack((np.dot(u, np.dot(W, v)).T, -u[:,2])).T,
           np.vstack((np.dot(u, np.dot(W.T, v)).T, u[:,2])).T,
           np.vstack((np.dot(u, np.dot(W.T, v)).T, -u[:,2])).T]
  return ret


