
import numpy as np
from skimage.feature import corner_harris, peak_local_max
import utils as ut
import matplotlib.pyplot as plt
import cv2 
from scipy.spatial.distance import cdist
import random
from skimage.color import rgb2gray



def get_harris_corners(im, edge_discard=20, min_distance=1):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    if im.shape[2] == 3:
        im = rgb2gray(im)

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=min_distance)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)

def anms_suspress(h, coords, c_robust=0.9, n_ip=500):
    distances = dist2(coords.T, coords.T)
    N = distances.shape[0]
    coords_y, coords_x = coords.T[:, 0], coords.T[:, 1]

    # Extract all corner strengths for our chosen corners in coords. h_coords[i] is the strength of the ith corner in coords.T
    h_coords = h[coords_y, coords_x].reshape(N, 1)

    # Collect resulting radii to sort later
    res_radii = np.full(N, np.inf)

    # From broadcasting, if stronger_mask[i][j] is True, then that means the jth corner is stronger then the ith corner by the robust threshold
    stronger_mask = h_coords < (c_robust * h_coords.T)

    for i in range(N):
        # Array where each element is the distance from corner i for the jth corner
        dist_to_i = distances[i, :]
        stronger_then_i = stronger_mask[i, :]
        if any(stronger_then_i):
            # save the radius neccesary for the closest neighbor that has strength stronger then i
            res_radii[i] = np.min(dist_to_i[stronger_then_i])
    

    sorted_indices = np.argsort(res_radii)[::-1] # sort in decreasing order
    best_corners = coords.T[sorted_indices][:n_ip] # grab the top n_ip corners
    return best_corners

def extract_feature_descriptors(img, coords):
    if img.shape[2] == 3:
        img = rgb2gray(img)

    N = coords.shape[0]
    PATCH_SIZE = 40
    W = PATCH_SIZE // 2
    SPACING = 5
    img_rows, img_cols = img.shape
    large_patches = []
    blurred_patches = []
    feature_descriptors = []
    for i in range(N):
        row, col = coords[i, 0], coords[i, 1]
        # if our patch does not go out of bounds
        if 0 <= row - W and row + W < img_rows and 0 <= col - W and col + W < img_cols:
            patch = img[row - W : row + W, 
                        col - W : col + W]
            large_patches.append(patch)
            
            # Blur the patch
            patch = cv2.GaussianBlur(patch, (5, 5), 0.5)
            blurred_patches.append(patch)


            # Crop the patch
            patch = patch[::SPACING, ::SPACING]

            # Normalize the patch to N(0, 1)
            patch_mean, patch_std = np.mean(patch), np.std(patch)

            z_patch = (patch - patch_mean) / (patch_std + 1e-6)
            feature_descriptors.append(z_patch)

    return large_patches, blurred_patches, feature_descriptors

def feature_match(img1, img2, coords1, coords2, threshold=0.7, n_ip=500):

    img1_descriptors = np.array(extract_feature_descriptors(img1, coords1)[2])
    img2_descriptors = np.array(extract_feature_descriptors(img2, coords2)[2])

    N1 = img1_descriptors.shape[0]
    N2 = img2_descriptors.shape[0]

    img1_descriptors = img1_descriptors.reshape((N1, 64))
    img2_descriptors = img2_descriptors.reshape((N2, 64))

    # Create a matrix of euclidean distances where dists[i][j] is the distance between img1_descriptors[i] and img2_descriptors[j]
    dists = cdist(img1_descriptors, img2_descriptors, 'euclidean')

    # Sort along columns. This means we will have the nearest neighbors sorted out for every descriptor in img1
    # First column is the 1NN, second is the 2NN, etc
    indices = np.argsort(dists, axis=1)

    matches_indices_1 = indices[:, 0] # 1NN
    matches_indices_2 = indices[:, 1] # 2NN


    dists_nn_1 = dists[np.arange(dists.shape[0]), matches_indices_1]
    dists_nn_2 = dists[np.arange(dists.shape[0]), matches_indices_2]

    ratios = dists_nn_1 / (dists_nn_2 + 1e-6)
    matches_mask = ratios < threshold
    
    query_indices = np.where(matches_mask)[0]
    nn_indices = matches_indices_1[matches_mask]

    return list(zip(query_indices, nn_indices))

def plot_matches(img1, img2, matches, coords1, coords2, mode="normal", n_ip=500, outpath=None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    canvas_height = max(h1, h2)
    canvas_width = w1 + w2
    canvas = np.zeros((canvas_height, canvas_width, 3))

    canvas[:h1, :w1, :] = img1
    canvas[:h2, w1:w1 + w2, :] = img2

    plt.figure(figsize=(20, 10))
    plt.imshow(canvas)
    plt.axis('off')

    for queryIdx, trainIdx in matches:
        pt1 = (int(coords1[queryIdx][1]), int(coords1[queryIdx][0]))
        pt2 = (int(coords2[trainIdx][1]), int(coords2[trainIdx][0]))

        color = (random.random(), random.random(), random.random())
        
        plt.plot(pt1[0], pt1[1], 'o', color=color, markersize=5)
        plt.plot(pt2[0] + w1, pt2[1], 'o', color=color, markersize=5)

        plt.plot([pt1[0], pt2[0] + w1], [pt1[1], pt2[1]], color=color, linewidth=1.5)

    plt.title(f"Found {len(matches)} Matches")
    if outpath:
        plt.savefig(f"final_output/{outpath}_{mode}_{n_ip}_correspondances.jpeg")
    plt.show()

def draw_matches(img1, img2, threshold=0.7, n_ip=500, mode="normal", outpath=None):
    # Extract gray img to compute initial correspondances

    # Compute initial correspondances
    h1, coords1 = get_harris_corners(img1, min_distance=15)
    h2, coords2 = get_harris_corners(img2, min_distance=15)
    coords1 = anms_suspress(h1, coords1, n_ip=n_ip)
    coords2 = anms_suspress(h2, coords2, n_ip=n_ip)

    # Get matches
    matches = feature_match(img1, img2, coords1, coords2, threshold=threshold, n_ip=n_ip)
    plot_matches(img1, img2, matches, coords1, coords2, mode=mode, n_ip=n_ip, outpath=outpath)

    


def plot_descriptor_extraction(large_patches, blurred_patches, feature_descriptors, n=5, outpath=None):

    if not large_patches:
        print("No valid patches were extracted to plot.")
        return

    n = min(n, len(large_patches))

    fig, axes = plt.subplots(3, n, figsize=(n * 3, 9.5))
    
    if n == 1:
        axes = axes.reshape(3, 1)

    axes[0, 0].set_ylabel('1. Original 40x40', fontsize=12, weight='bold')
    axes[1, 0].set_ylabel('2. Blurred 40x40', fontsize=12, weight='bold')
    axes[2, 0].set_ylabel('3. 8x8 Descriptor', fontsize=12, weight='bold')

    for i in range(n):
        axes[0, i].imshow(large_patches[i], cmap='gray')
        axes[0, i].set_title(f"Corner #{i+1}")
        axes[0, i].axis('off')

        axes[1, i].imshow(blurred_patches[i], cmap='gray')
        axes[1, i].axis('off')

        axes[2, i].imshow(feature_descriptors[i], cmap='gray')
        axes[2, i].axis('off')
        
    fig.suptitle("Feature Descriptor Extraction Steps", fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    if outpath is not None:
        plt.savefig(f"final_output/{outpath}_descriptors")
    plt.show()




# Plots corners on img. If n_ip is specified then anms suspression is used
def plot_corners(img, outpath, min_distance=1, n_ip=None, c_robust=0.9):
    h, coords = get_harris_corners(img, min_distance=min_distance)
    plt.imshow(img)

    if n_ip is not None:
        coords = anms_suspress(h, coords, c_robust=c_robust, n_ip=n_ip).T

    plt.plot(coords[1], coords[0], '+r', markersize=5) 
    plt.savefig(f"final_output/{outpath}_corners")
    plt.show()
    plt.clf()

# Testing Functions to Reproduce Results
def test_harris_corners():
    park_img_1 = ut.read_in_image("data/images/park_1.jpeg")
    plot_corners(park_img_1, outpath="park_1_harris", min_distance=15)
    plot_corners(park_img_1, outpath="park_1_harris_suspress_250", min_distance=15, n_ip=250)
    plot_corners(park_img_1, outpath="park_1_harris_suspress_125", min_distance=15, n_ip=125)


def test_feature_extraction():
    park_img_1 = ut.read_in_image("data/images/park_1.jpeg")
    h, coords = get_harris_corners(park_img_1, min_distance=15)
    coords = anms_suspress(h, coords, n_ip=500)
    large_patches, blurred_patches, descriptors = extract_feature_descriptors(park_img_1, coords)
    plot_descriptor_extraction(large_patches[20:], blurred_patches[20:], descriptors[20:], n=5, outpath="park_1_500")

    berkeley_img_1 = ut.read_in_image("data/images/berkeley_3.jpeg")
    h, coords = get_harris_corners(berkeley_img_1, min_distance=15)
    coords = anms_suspress(h, coords, n_ip=500)
    large_patches, blurred_patches, descriptors = extract_feature_descriptors(berkeley_img_1, coords)
    plot_descriptor_extraction(large_patches[20:], blurred_patches[20:], descriptors[20:], n=5, outpath="berkeley_1_500")

def test_feature_matching():
    park_img_1 = ut.read_in_image("data/images/park_1.jpeg")
    park_img_2 = ut.read_in_image("data/images/park_2.jpeg")
    h1, coords1 = get_harris_corners(park_img_1, min_distance=15)
    h2, coords2 = get_harris_corners(park_img_2, min_distance=15)
    coords1 = anms_suspress(h1, coords1, n_ip=500)
    coords2 = anms_suspress(h2, coords2, n_ip=500)
    draw_matches(park_img_1, park_img_2, threshold=0.7, n_ip=500, outpath="park_1_to_2_500")

    berkeley_img_1 = ut.read_in_image("data/images/berkeley_3.jpeg")
    berkeley_img_2 = ut.read_in_image("data/images/berkeley_4.jpeg")
    h1, coords1 = get_harris_corners(berkeley_img_1, min_distance=15)
    h2, coords2 = get_harris_corners(berkeley_img_2, min_distance=15)
    coords1 = anms_suspress(h1, coords1, n_ip=500)
    coords2 = anms_suspress(h2, coords2, n_ip=500)
    draw_matches(berkeley_img_1, berkeley_img_2, threshold=0.7, n_ip=500, outpath="berkeley_1_to_2_500")

if __name__ == "__main__":
    test_harris_corners()
    test_feature_extraction()
    test_feature_matching()


