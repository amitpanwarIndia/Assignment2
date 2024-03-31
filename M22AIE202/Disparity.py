import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

I1 = cv2.imread('bikeL.png')
I2 = cv2.imread('bikeR.png')

# Loaded matrices and baseline from bike.txt
cam0 = np.array([[5299.313, 0, 1263.818], [0, 5299.313, 977.763], [0, 0, 1]])
cam1 = np.array([[5299.313, 0, 1438.004], [0, 5299.313, 977.763], [0, 0, 1]])
baseline = 177.288

sift = cv2.SIFT_create()

# keypoints and descriptors calculation
kp1, des1 = sift.detectAndCompute(I1, None)
kp2, des2 = sift.detectAndCompute(I2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# disparity and depth maps calculation
disparity_map = np.zeros_like(I1[:, :, 0], dtype=np.float32)
depth_map = np.zeros_like(I1[:, :, 0], dtype=np.float32)

for m in good:
    query_idx = m.queryIdx
    train_idx = m.trainIdx

    u1, v1 = kp1[query_idx].pt
    u2, v2 = kp2[train_idx].pt

    # disparity
    disparity = np.abs(u1 - u2)
    disparity_map[int(v1), int(u1)] = disparity

    # depth using the average of both cameras
    focal_length = (cam0[0, 0] + cam1[0, 0]) / 2
    depth = (focal_length * baseline) / disparity
    depth_map[int(v1), int(u1)] = depth

# Normalize disparity
disparity_map_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow('Disparity Map', disparity_map_normalized)
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create 3D point cloud
h, w = depth_map.shape
y, x = np.mgrid[:h, :w]
cloud = np.stack((x, y, depth_map), axis=-1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(cloud[:, :, 0], cloud[:, :, 1], cloud[:, :, 2])
plt.show()
