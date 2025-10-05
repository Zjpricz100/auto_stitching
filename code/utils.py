import os
import skimage.io as skio
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from skimage import img_as_float
from PIL import Image, ImageOps


def write_output(image, imname, outpath="final_output"):
    os.makedirs(outpath, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(imname))[0]
    out_path = os.path.join(outpath, f"{base_name}.jpg")
    plt.imsave(out_path, image, cmap='gray')

    print("Saved to ", out_path)


def min_max_normalize_image(image):
    display = np.zeros(image.shape)

    if image.ndim == 3:
        for c in range(image.shape[2]):
            ch = image[..., c]
            display[..., c] = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)
    else:
        display = (image - image.min()) / (image.max() - image.min() + 1e-8)

    return display

def crop(img, top_percent=0.0, bottom_percent=0.0, left_percent=0.0, right_percent=0.0):
    H, W = img.shape[:2]

    # Convert percents into pixel counts (rounded to int)
    top = int(H * top_percent)
    bottom = int(H * bottom_percent)
    left = int(W * left_percent)
    right = int(W * right_percent)

    return img[top:H - bottom if bottom > 0 else H,
               left:W - right if right > 0 else W]


def read_in_image(imname, gray=False, plot=False):
    # Load with Pillow and fix orientation
    img = Image.open(imname)
    img = ImageOps.exif_transpose(img)
    im = np.array(img)

    im = img_as_float(im)

    if im.ndim == 3 and im.shape[2] > 3:
        im = im[:, :, :3]

    if gray and im.ndim == 3:
        im = rgb2gray(im)

    if plot:
        plt.imshow(im, cmap="gray" if gray else None)
        plt.show()

    return im

# Util functions for loading correspondances:
def get_correspondances(path, n):


    # Load the points from the file
    data = np.load(f'data/points/{path}')

    # Access the arrays
    points1 = data['points1']
    points2 = data['points2']

    if points1.shape[0] < n or points1.shape[0] != points2.shape[0]:
        print("ERROR: not enough saved correspondances for requested amount.")
        return

    return points1[:n], points2[:n]
    #print("Points from the first image:\n", points1)
    #print("\nPoints from the second image:\n", points2)

    # Now you can use these points with libraries like OpenCV to find the homography
    # e.g., homography_matrix, mask = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)