import numpy as np  
import utils as ut # File utility functions \
import matplotlib.pyplot as plt
import math

def get_img_correspondances(correspondance_path, n_correspondances):
    points1, points2 = ut.get_correspondances(correspondance_path, n=n_correspondances)
    assert points1.shape[0] == points2.shape[0] == n_correspondances
    return points1, points2



def setup_homography(img1_points, img2_points):
    """
    Computes the data matrix D for Dh = b' and also returns the vector b
    """

    n_correspondances = img1_points.shape[0]
    D = []
    


    for i in range(n_correspondances):
        x, y = img1_points[i]
        u, v = img2_points[i]

        D_row_1 = np.array([x, y, 1, 0, 0, 0, -u*x, -u*y])
        D_row_2 = np.array([0, 0, 0, x, y, 1, -v*x, -v*y])
        D.append(D_row_1)
        D.append(D_row_2)

    assert len(D) == n_correspondances * 2
    return np.array(D), img2_points.reshape(-1)
    


def compute_homography(img1_points, img2_points):
    D, b = setup_homography(img1_points, img2_points)

    # Use least squares to compute the best h according to our data matrix D
    h = np.linalg.lstsq(D, b)[0]

    # Recover homography matrix H. Fix last component to 0
    h = np.concat([h, [1]])
    H = h.reshape((3, 3))
    return H

def warp_image_nearest_neighbor(source_img, reference_img, H):
    out_H, out_W = reference_img.shape[0], reference_img.shape[1]
    in_H, in_W = source_img.shape[0], source_img.shape[1]
    output_img = np.zeros_like(reference_img)
    

    H_inv = np.linalg.inv(H)
    for y_prime in range(out_H):
        for x_prime in range(out_W):

            src_coordinates_homogeneous = H_inv @ (np.array([x_prime, y_prime, 1]))
            w = src_coordinates_homogeneous[2]

            # Normalize by the depth
            x, y = src_coordinates_homogeneous[0] / w, src_coordinates_homogeneous[1] / w

            # Valid coordinates to interpolate. Otherwise we skip and leave black
            # We interpolate using the corners as rays implementation
            x_source, y_source = round(x), round(y)
            if 0 <= x_source < in_W and 0 <= y_source < in_H:
                output_img[y_prime][x_prime] = source_img[y_source][x_source]

    return output_img




def rectify(source_img, img1_points):
    """
    Rectifies the rectange from the first image points to a flat 2D projected rectangle maintaining same aspect ratio
    
    Assume img1_points is in order of top_left, top_right, bottom_left, bottom_right
    """
    assert img1_points.shape[0] == 4
    tl, tr, bl, br = img1_points
    
    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    width = round(max(width_top, width_bottom))

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    height = round(max(height_left, height_right))

    reference_img = np.zeros((height, width, 3))
    img2_points = np.array([(0,0), (width-1, 0), (0, height-1), (width-1, height-1)])
    
    H = compute_homography(img1_points, img2_points)
    return warp_image_nearest_neighbor(source_img, reference_img, H)
    
def rectify_img(im_name):
    points = get_img_correspondances(f"{im_name}_rect.npz", n_correspondances=4)[0]
    img = ut.read_in_image(f"data/images/{im_name}.jpeg")
    rectified_img = rectify(img, points)
    ut.write_output(rectified_img, f"{im_name}_rectified.jpeg")
rectify_img("code_names")





