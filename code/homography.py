import numpy as np  
import utils as ut # File utility functions \
import matplotlib.pyplot as plt
import math

from scipy.ndimage import distance_transform_edt


def get_img_correspondances(correspondance_path, n_correspondances):
    points1, points2 = ut.get_correspondances(correspondance_path, n=n_correspondances)
    assert points1.shape[0] == points2.shape[0] == n_correspondances
    return points1, points2

# HOMOGRAPHIES

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

# IMAGE WARPING AND INTERPOLATION

def warp_image_nearest_neighbor(source_img, reference_img, H, rectify=False, output_img=None, offset_x=None, offset_y=None, source_mask=None):
    in_H, in_W = source_img.shape[0], source_img.shape[1]

    if not rectify:
        source_corners = [(0,0), (in_W-1, 0), (0, in_H-1), (in_W-1, in_H-1)]
        warped_corners = compute_warped_corners(source_corners, H)
        all_corners = np.vstack((source_corners, warped_corners))
        min_x, min_y = np.min(all_corners, axis=0)
        max_x, max_y = np.max(all_corners, axis=0)

        out_H = int(abs(max_y - min_y))
        out_W = int(abs(max_x - min_x))
        output_img = np.zeros((out_H, out_W, 3))
    else:
        out_H, out_W = reference_img.shape[0], reference_img.shape[1]
        output_img = np.zeros_like(reference_img)
        min_x, min_y = 0, 0
        

    H_inv = np.linalg.inv(H)
    for y_prime in range(out_H):
        for x_prime in range(out_W):

            # adding offset
            global_x = x_prime + min_x
            global_y = y_prime + min_y


            src_coordinates_homogeneous = H_inv @ (np.array([global_x, global_y, 1]))
            w = src_coordinates_homogeneous[2]

            # Normalize by the depth
            x, y = src_coordinates_homogeneous[0] / w, src_coordinates_homogeneous[1] / w

            # Valid coordinates to interpolate. Otherwise we skip and leave black
            # We interpolate using the corners as rays implementation
            x_source, y_source = round(x), round(y)
            if 0 <= x_source < in_W and 0 <= y_source < in_H:
                output_img[y_prime][x_prime] = source_img[y_source][x_source]

    return output_img

def warp_image_bilinear(source_img, reference_img, H, rectify=False, output_img=None, offset_x=None, offset_y=None, source_mask=None):
    in_H, in_W = source_img.shape[0], source_img.shape[1]

    # If we dont provide an output img we need to compute it with appropiate offsets
    if output_img is None:
        if not rectify:
            source_corners = [(0,0), (in_W-1, 0), (0, in_H-1), (in_W-1, in_H-1)]
            warped_corners = compute_warped_corners(source_corners, H)
            all_corners = np.vstack((source_corners, warped_corners))
            min_x, min_y = np.min(all_corners, axis=0)
            max_x, max_y = np.max(all_corners, axis=0)

            out_H = int(abs(max_y - min_y))
            out_W = int(abs(max_x - min_x))
            output_img = np.zeros((out_H, out_W, 3))
        else:
            out_H, out_W = reference_img.shape[0], reference_img.shape[1]
            output_img = np.zeros_like(reference_img)
            min_x, min_y = 0, 0
    else:
        min_x, min_y = offset_x, offset_y
        out_H, out_W = output_img.shape[:2]
            

    H_inv = np.linalg.inv(H)
    for y_prime in range(out_H):
        for x_prime in range(out_W):

            # adding offset
            global_x = x_prime + min_x
            global_y = y_prime + min_y


            src_coordinates_homogeneous = H_inv @ (np.array([global_x, global_y, 1]))
            w = src_coordinates_homogeneous[2]

            # Normalize by the depth
            x, y = src_coordinates_homogeneous[0] / w, src_coordinates_homogeneous[1] / w

            if 1 <= x < in_W - 1 and 1 <= y < in_H - 1:
                # Bilinear interpolation: Weighted average of neighboring pixels
                x_floor, y_floor = int(x), int(y)
                dx = x - x_floor
                dy = y - y_floor

                p_tl = source_img[y_floor, x_floor]       # Top-left
                p_tr = source_img[y_floor, x_floor + 1]   # Top-right
                p_bl = source_img[y_floor + 1, x_floor]   # Bottom-left
                p_br = source_img[y_floor + 1, x_floor + 1] # Bottom-right

                # Interpolate horizontally (along the x-axis)
                top_interp = (1 - dx) * p_tl + dx * p_tr
                bottom_interp = (1 - dx) * p_bl + dx * p_br

                # Interpolate vertically (along the y-axis)
                final_pixel = (1 - dy) * top_interp + dy * bottom_interp
                
                output_img[y_prime, x_prime] = final_pixel.astype(source_img.dtype)

                # If a source mask was provided, fill that pixel too for the valid interpolation
                if source_mask is not None:
                    source_mask[y_prime, x_prime] = 1.0

    return output_img

# RECTIFICATION

# Returns TL, TR, BL, BR

def compute_warped_corners(img_points, H):
    img_points_homogeneous = np.column_stack((img_points, np.ones(len(img_points)))).T  # shape (3, N)
    warped = H @ img_points_homogeneous  # shape (3, N)
    warped /= warped[2, :]
    warped_corners = warped[:2, :].T
    return warped_corners

def rectify(source_img, img1_points):
    """
    Rectifies the rectange from the first image points to a flat 2D projected rectangle maintaining same aspect ratio
    
    Assume img1_points is in order of top_left, top_right, bottom_left, bottom_right
    """
    assert img1_points.shape[0] == 4
    tl, tr, bl, br = img1_points

    width_top = np.linalg.norm(tl - tr)
    width_bottom = np.linalg.norm(bl - br)
    avg_width = (width_top + width_bottom) / 2

    height_left = np.linalg.norm(tl - bl)
    height_right = np.linalg.norm(tr - br)
    avg_height = (height_left + height_right) / 2

    width, height = int(avg_width), int(avg_height)

    img2_points = np.array([(0,0), (width-1, 0), (0, height-1), (width-1, height-1)])
    
    H = compute_homography(img1_points, img2_points)
    img1_points_homogeneous = np.column_stack((img1_points, np.ones(len(img1_points)))).T  # shape (3, N)
    warped = H @ img1_points_homogeneous  # shape (3, N)
    warped /= warped[2, :]
    warped_corners = warped[:2, :].T

    xs = warped_corners[:, 0]
    ys = warped_corners[:, 1]
    min_x, max_x = int(np.floor(xs.min())), int(np.ceil(xs.max()))
    min_y, max_y = int(np.floor(ys.min())), int(np.ceil(ys.max()))
    width, height = max_x - min_x, max_y - min_y

    reference_img = np.zeros((height, width, 3))
 
    return warp_image_nearest_neighbor(source_img, reference_img, H, rectify=True)
    
def rectify_img(im_name, out_path=""):
    points = get_img_correspondances(f"{im_name}_rect.npz", n_correspondances=4)[0]
    img = ut.read_in_image(f"data/images/{im_name}.jpeg")
    rectified_img = rectify(img, points)
    ut.write_output(rectified_img, f"{im_name}_{out_path}_rectified.jpeg")

def warp_img(source_img, reference_img, source_points, reference_points, out_path=""):
    H = compute_homography(source_points, reference_points)
    print(H)
    warped_source = warp_image_bilinear(source_img, reference_img, H)
    ut.write_output(warped_source, f"{out_path}.jpeg")

# CREATING MOSAICS    

def stitch_images(source_img, reference_img, source_points, reference_points):
    print("Computing Homography...")
    H = compute_homography(source_points, reference_points)

    source_H, source_W = source_img.shape[:2]
    ref_H, ref_W = reference_img.shape[:2]

    # Calculate a bounding box that can support both the warped source and reference img
    source_corners = np.array([[0,0], [source_W-1, 0], [0, source_H-1], [source_W-1, source_H-1]])
    reference_corners = np.array([[0,0], [ref_W-1, 0], [0, ref_H-1], [ref_W-1, ref_H-1]])
    warped_corners = compute_warped_corners(source_corners, H)

    all_corners = np.vstack((reference_corners, warped_corners))
    min_x, min_y = np.floor(np.min(all_corners, axis=0)).astype(int)
    max_x, max_y = np.ceil(np.max(all_corners, axis=0)).astype(int)

    out_H = max_y - min_y + 1
    out_W = max_x - min_x + 1

    # Creating output image and masks
    output_img = np.zeros((out_H, out_W, 3))
    source_mask = np.zeros((out_H, out_W, 3))
    reference_mask = np.zeros((out_H, out_W, 3))


    # Place the reference image onto the new canvas, with the accounted for offset. Do the same for reference mask
    reference_mask[-min_y : ref_H - min_y, -min_x : ref_W - min_x, :] = 1.0
    output_img[-min_y : ref_H - min_y, -min_x : ref_W - min_x, :] = reference_img


    # Warp the source img onto this new canvas, also warp the source mask the same way
    print("Warping Source Image onto Reference Img with H...")
    output_img = warp_image_bilinear(source_img, reference_img, H, rectify=False, output_img=output_img, offset_x=min_x, offset_y=min_y, source_mask=source_mask)

    # Create smooth gradient blending masks using distance transform. Normalize them.
    print("Creating Blending Masks...")
    alpha_source = distance_transform_edt(source_mask)
    alpha_source = alpha_source / (alpha_source.max() or 1)

    alpha_reference = distance_transform_edt(reference_mask)
    alpha_reference = alpha_reference / (alpha_reference.max() or 1)

    # Final blending mask
    blending_mask = alpha_source / (alpha_source + alpha_reference + 1e-8)

    # Weighted averaging of pixels with masks
    blended_img = np.zeros_like(output_img)
    blended_img[-min_y : ref_H - min_y, -min_x : ref_W - min_x, :] = reference_img

    blended_img = blending_mask * output_img + (1 - blending_mask) * blended_img

    return blended_img

def create_multiple_img_mosaic(images, points, center_idx, out_path):


    print("Computing Homographies and Final Bounding Box...")
    homographies = []
    all_corners = [] # Holds the warped corners
    points_idx = 0
    for idx, image in enumerate(images):

        H, W = image.shape[:2]
        corners = np.array([[0, 0], [W-1, 0], [0, H-1], [W-1, H-1]])

        # If we are at our reference image, we just apply identity to it as our homography
        if idx == center_idx:
            H = np.eye(3)
            all_corners.append(corners)
        else:
            correspondances = points[points_idx]
            H = compute_homography(correspondances[0], correspondances[1])
            warped_corners = compute_warped_corners(corners, H)
            all_corners.append(warped_corners)
            points_idx += 1

        homographies.append(H)

    all_corners = np.vstack(all_corners)
    min_x, min_y = np.floor(np.min(all_corners, axis=0)).astype(int)
    max_x, max_y = np.ceil(np.max(all_corners, axis=0)).astype(int)
    out_H = max_y - min_y + 1
    out_W = max_x - min_x + 1

    print("Warping Every Image with its Appropiate Homography...")
    warped_layers = []
    mask_layers = []
    reference_img = images[center_idx]

    for idx, image in enumerate(images):

        # Creating output image and masks
        output_img = np.zeros((out_H, out_W, 3))
        source_mask = np.zeros((out_H, out_W, 3))
        
        warped_layer = warp_image_bilinear(images[idx], reference_img, homographies[idx], rectify=False, output_img=output_img, offset_x=min_x, offset_y=min_y, source_mask=source_mask)

        warped_layers.append(warped_layer)
        mask_layers.append(source_mask)

    print("Creating Blending Masks...")
    alpha_masks = []
    for mask in mask_layers:
        alpha_source = distance_transform_edt(mask)
        alpha_source = alpha_source / (alpha_source.max() or 1)
        alpha_masks.append(alpha_source)

    total_alpha = np.sum(alpha_masks, axis=0) + 1e-8
    blended_img = np.zeros((out_H, out_W, 3))

    print("Blending Final Image Together...")
    for i in range(len(images)):
        weight = alpha_masks[i] / total_alpha
        print(weight.shape, warped_layers[i].shape, blended_img.shape)

        blended_img += (weight * warped_layers[i])

    ut.write_output(blended_img, f"{out_path}_mosaic.jpg")







    # for idx, image in enumerate(images):


    #     reference_mask[-min_y : ref_H - min_y, -min_x : ref_W - min_x, :] = 1.0
    #     output_img[-min_y : ref_H - min_y, -min_x : ref_W - min_x, :] = reference_img





def create_moasaic(images, points, out_path=""):
    assert len(images) >= 2
    assert len(points) == len(images) - 1
    n_images = len(images)
    i = 0
    current_img = images[i]
    while i < n_images - 1:
        next_img = images[i + 1]
        next_points = points[i]
        current_img = stitch_images(current_img, next_img, next_points[0], next_points[1])
    ut.write_output(current_img, f"out_path_mosaic.jpeg")



def create_img_mosaic(left_img, right_img, left_points, right_points, out_path=""):
    blended_img = stitch_images(left_img, right_img, left_points, right_points)
    ut.write_output(blended_img, f"{out_path}_mosaic.jpeg")



#rectify_img("skull")

park_1_img = ut.read_in_image("data/images/park_1.jpeg")
park_2_img = ut.read_in_image("data/images/park_2.jpeg")
park_3_img = ut.read_in_image("data/images/park_3.jpeg")
park_4_img = ut.read_in_image("data/images/park_4.jpeg")
park_images = [park_1_img, park_2_img, park_3_img, park_4_img]
park_1_to_2_points = get_img_correspondances("park_1_to_2.npz", n_correspondances=20)
park_2_to_3_points = get_img_correspondances("park_2_to_3.npz", n_correspondances=20)
park_4_to_2_points = get_img_correspondances("park_4_to_2.npz", n_correspondances=20)
park_points = [park_1_to_2_points, (park_2_to_3_points[1], park_2_to_3_points[0]), park_4_to_2_points]

stairs_1_img = ut.read_in_image("data/images/stairs_1.jpeg")
stairs_2_img = ut.read_in_image("data/images/stairs_2.jpeg")
stairs_3_img = ut.read_in_image("data/images/stairs_3.jpeg")
stairs_images = [stairs_1_img, stairs_2_img, stairs_3_img]
stairs_1_to_2_points = get_img_correspondances("stairs_1_to_2.npz", n_correspondances=20)
stairs_3_to_2_points = get_img_correspondances("stairs_3_to_2.npz", n_correspondances=20)
stairs_points = [stairs_1_to_2_points, stairs_3_to_2_points]

berkeley_3_img = ut.read_in_image("data/images/berkeley_3.jpeg")
berkeley_4_img = ut.read_in_image("data/images/berkeley_4.jpeg")
berkeley_5_img = ut.read_in_image("data/images/berkeley_5.jpeg")
berkeley_images = [berkeley_3_img, berkeley_4_img, berkeley_5_img]

berkeley_3_to_4_points = get_img_correspondances("berkeley_3_to_4.npz", n_correspondances=20)
berkeley_5_to_4_points = get_img_correspondances("berkeley_5_to_4.npz", n_correspondances=20)
berkeley_points = [berkeley_3_to_4_points, berkeley_5_to_4_points]




#park_mosaic = create_moasaic(park_images, park_points, "park")

#create_img_mosaic(park_1_img, park_2_img, park_1_to_2_points[0], park_1_to_2_points[1])

#create_multiple_img_mosaic(park_images, park_points, center_idx=1, out_path="park_multiple")
#create_multiple_img_mosaic(stairs_images, stairs_points, center_idx=1, out_path="stairs")
#create_multiple_img_mosaic(berkeley_images, berkeley_points, center_idx=1, out_path="berkeley")



warp_img(park_2_img, park_3_img, park_2_to_3_points[0], park_2_to_3_points[1], out_path="park_2_to_3.jpeg")

#rectify_img("skull", out_path="nearest")




