import cv2
import numpy as np

def draw_epipolar_lines(img1, img2, F):
    img1_with_lines = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2_with_lines = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    corresponding_pts_img1 = []
    corresponding_pts_img2 = []

    # Epipolar lines for first image
    for i in range(0, img1.shape[0], 10):
        x = np.array([0, i, 1])
        line = np.dot(F, x)

        # find two points on the epipolar line in image 2
        pt1_img2 = np.array([0, -line[2] / line[1]])
        pt2_img2 = np.array([img2.shape[1], -(line[2] + line[0] * img2.shape[1]) / line[1]])

        # equation of the line
        seg_length = np.linalg.norm(pt2_img2 - pt1_img2)
        seg_unit_vector = (pt2_img2 - pt1_img2) / seg_length

        # uniformly spaced 10 pixels along the line segment
        for j in range(0, int(seg_length), 10):
            pt = pt1_img2 + j * seg_unit_vector
            corresponding_pts_img1.append(pt)

        # Draw line segment on image 2
        cv2.line(img2_with_lines, (int(pt1_img2[0]), int(pt1_img2[1])), (int(pt2_img2[0]), int(pt2_img2[1])), (0, 255, 0), 1)

    # Epipolar lines for second image
    for i in range(0, img2.shape[0], 10):
        x = np.array([0, i, 1])
        line = np.dot(F.T, x)

        # finding two points on the epipolar line in image 1
        pt1_img1 = np.array([0, -line[2] / line[1]])
        pt2_img1 = np.array([img1.shape[1], -(line[2] + line[0] * img1.shape[1]) / line[1]])

        # Find the equation of the line
        seg_length = np.linalg.norm(pt2_img1 - pt1_img1)
        seg_unit_vector = (pt2_img1 - pt1_img1) / seg_length

        # uniformly spaced 10 pixels along the line segment
        for j in range(0, int(seg_length), 10):
            pt = pt1_img1 + j * seg_unit_vector
            corresponding_pts_img2.append(pt)

        # Draw line segment on image 1
        cv2.line(img1_with_lines, (int(pt1_img1[0]), int(pt1_img1[1])), (int(pt2_img1[0]), int(pt2_img1[1])), (0, 255, 0), 1)

    cv2.imshow('Image 1 with epipolar lines', img1_with_lines)
    cv2.imshow('Image 2 with epipolar lines', img2_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return corresponding_pts_img1, corresponding_pts_img2

img1 = cv2.imread('000000.png', 0)
img2 = cv2.imread('000023.png', 0)

# Fundamental matrix from FM.txt
F = np.array([[3.34638533e-07,  7.58547151e-06, -2.04147752e-03],
              [-5.83765868e-06,  1.36498636e-06, 2.67566877e-04],
              [1.45892349e-03, -4.37648316e-03,  1.00000000e+00]])

corresponding_pts_img1, corresponding_pts_img2 = draw_epipolar_lines(img1, img2, F)

print("Corresponding points in image 1:")
print(corresponding_pts_img1)
print("Corresponding points in image 2:")
print(corresponding_pts_img2)
