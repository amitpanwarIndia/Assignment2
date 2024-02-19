import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import os
from scipy.spatial import distance
import cv2
###############################################################################################################################
################################ Affine transformation ########################################################################
###############################################################################################################################

def affine_transform(image, scale=1.0, tx=0, ty=0, theta=0, shear=0):
    transform = np.array([
        [scale*np.cos(theta), -shear, tx],
        [scale*np.sin(theta), scale, ty]
    ])

    h, w = image.shape
    y, x = np.mgrid[:h, :w]
    ones = np.ones_like(x)
    coords = np.stack([x, y, ones])
    coords = coords.reshape(3, -1)
    new_coords = np.dot(transform, coords)
    new_coords = new_coords.reshape(2, h, w)
    new_image = map_coordinates(image, new_coords, order=1)

    return new_image

image = Image.open('../pattern.png').convert('L')
image = np.array(image)

transformed_image = affine_transform(image, scale=1.2, tx=30, ty=50, theta=np.pi/6, shear=0.1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(transformed_image, cmap='gray')
plt.title('Transformed Image')
plt.show()

###############################################################################################################################
################################ Ripple transformation ########################################################################
###############################################################################################################################

def ripple_transform(image, amplitude=10, frequency=0.1):
    h, w = image.shape
    y, x = np.mgrid[:h, :w]
    dx = amplitude * np.sin(2 * np.pi * frequency * y / h)
    x_new = x + dx
    x_new = np.clip(x_new, 0, w - 1)
    new_image = map_coordinates(image, [y, x_new], order=1)

    return new_image

image = Image.open('../pattern.png').convert('L')
image = np.array(image)

transformed_image = ripple_transform(image, amplitude=80, frequency=0.5)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(transformed_image, cmap='gray')
plt.title('Ripple Transformed Image')
plt.show()


###############################################################################################################################
################################ Harris corner detection technique ############################################################
###############################################################################################################################

def harris_corner_detection(image, window_size=3, k=0.04, threshold=0.01):
    Ix = np.gradient(image.astype(np.float64), axis=1)
    Iy = np.gradient(image.astype(np.float64), axis=0)
    Ixx = Ix ** 2
    Ixy = Ix * Iy
    Iyy = Iy ** 2

    weight = np.ones((window_size, window_size)) / (window_size ** 2)

    Ixx = Ixx.reshape(image.shape)
    Ixy = Ixy.reshape(image.shape)
    Iyy = Iyy.reshape(image.shape)

    Sxx = convolve2d(Ixx, weight, mode='same')
    Sxy = convolve2d(Ixy, weight, mode='same')
    Syy = convolve2d(Iyy, weight, mode='same')

    det = Sxx * Syy - Sxy ** 2
    trace = Sxx + Syy
    R = det - k * (trace ** 2)

    corners = np.zeros_like(R)
    corners[R > threshold * R.max()] = 255

    return corners


def load_images(folder_path):
    images = []
    orig_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".JPEG"):
            image = np.array(Image.open(os.path.join(folder_path, filename)).convert('L'))
            images.append(image)

            orig_image = np.array(Image.open(os.path.join(folder_path, filename)))
            orig_images.append(orig_image)
    return images, orig_images


folder_path = "../images_2" 
images, orig_images = load_images(folder_path)

corner_images = []
for image in images:
    corners = harris_corner_detection(image, window_size=8, threshold=0.001)
    corner_images.append(corners)

# img_path = '../images_2/1.jpeg'
# orig_image = np.array(Image.open(img_path))
# converted_image = np.array(Image.open(img_path).convert('L'))

# corners = harris_corner_detection(converted_image, window_size=5, threshold=0.005)

plt.figure(figsize=(30, 20))
num_images = len(images)
for i in range(num_images):
    plt.subplot(2, num_images, i+1)
    plt.imshow(orig_images[i])
    plt.title('Image {}'.format(i+1))
    plt.axis('off')

    plt.subplot(2, num_images, num_images + i + 1)
    plt.imshow(corner_images[i], cmap='gray')
    plt.title('Corners {}'.format(i+1))
    plt.axis('off')

plt.show()


###############################################################################################################################
################################ Panorama using Image Stitching ###############################################################
###############################################################################################################################
def match_features(descriptors1, descriptors2):

    dist_matrix = distance.cdist(descriptors1, descriptors2, 'euclidean')
    matches = []
    for i in range(dist_matrix.shape[0]):
        min_idx = np.argmin(dist_matrix[i])
        matches.append((i, min_idx))
    return matches

def ransac_homography(matches, keypoints1, keypoints2, num_iterations=100, threshold=5):
    best_H = None
    max_inliers = 0

    for _ in range(num_iterations):

        sample_indices = np.random.choice(len(matches), 4, replace=False)
        sample_matches = [matches[i] for i in sample_indices]

        src_pts = np.float32([keypoints1[m[0]].pt for m in sample_matches])
        dst_pts = np.float32([keypoints2[m[1]].pt for m in sample_matches])
     
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
   
        inliers = 0
        for match in matches:
            src_pt = keypoints1[match[0]].pt
            dst_pt = keypoints2[match[1]].pt
            src_pt_hom = np.array([src_pt[0], src_pt[1], 1]).reshape(3, 1)
            transformed_pt = np.dot(H, src_pt_hom)
            transformed_pt /= transformed_pt[2]  # Normalize
            if np.linalg.norm(transformed_pt[:2] - dst_pt) < threshold:
                inliers += 1
   
        if inliers > max_inliers:
            best_H = H
            max_inliers = inliers

    return best_H

def warp_image(image, homography):
    if homography is None:
        return image.copy() 
    else:
        output_image = np.zeros_like(image)
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                transformed_pt = np.dot(homography, [x, y, 1])
                x_out, y_out = transformed_pt[0] / transformed_pt[2], transformed_pt[1] / transformed_pt[2]
                output_image[y, x] = bilinear_interpolation(image, x_out, y_out)
        return output_image


def bilinear_interpolation(image, x, y):
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + 1, y1 + 1

    x1, y1 = max(0, min(x1, image.shape[1] - 1)), max(0, min(y1, image.shape[0] - 1))
    x2, y2 = max(0, min(x2, image.shape[1] - 1)), max(0, min(y2, image.shape[0] - 1))

    dx, dy = x - x1, y - y1

    interpolated_value = (1 - dx) * (1 - dy) * image[y1, x1] + \
                         dx * (1 - dy) * image[y1, x2] + \
                         (1 - dx) * dy * image[y2, x1] + \
                         dx * dy * image[y2, x2]

    return interpolated_value

def blend_images(image1, image2):

    return (image1 + image2) / 2

image1 = cv2.imread('../images_1/image1.jpg')
image2 = cv2.imread('../images_1/image2.jpg')
image3 = cv2.imread('../images_1/image3.jpg')

gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
gray_image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)
keypoints3, descriptors3 = sift.detectAndCompute(gray_image3, None)

image1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None)
image2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None)
image3_with_keypoints = cv2.drawKeypoints(image3, keypoints3, None)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image1_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Image 1 with Keypoints')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(image2_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Image 2 with Keypoints')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(image3_with_keypoints, cv2.COLOR_BGR2RGB))
plt.title('Image 3 with Keypoints')
plt.axis('off')

plt.show()

matches1_2 = match_features(descriptors1, descriptors2)
matches2_3 = match_features(descriptors2, descriptors3)

homography1_2 = ransac_homography(matches1_2, keypoints1, keypoints2)
homography2_3 = ransac_homography(matches2_3, keypoints2, keypoints3)

warped_image2 = warp_image(image2, homography1_2)
warped_image3 = warp_image(image3, homography2_3)

blended_image = blend_images(image1, blend_images(warped_image2, warped_image3))

plt.figure(figsize=(12, 6))
plt.imshow(blended_image)
plt.title('Stitched Panorama')
plt.axis('off')
plt.show()
