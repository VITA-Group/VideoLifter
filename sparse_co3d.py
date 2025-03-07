import torch
import numpy as np
from scipy.optimize import least_squares
import cv2
import open3d as o3d
import os
import matplotlib.pyplot as plt
import math
import time
import pickle
import argparse
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "mast3r")))

from utils.dust3r_utils import load_images, compute_global_alignment
from mast3r.model import AsymmetricMASt3R
from mast3r.fast_nn import fast_reciprocal_NNs
from dust3r.inference import inference
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy

def reprojection_error(params, points_3d, points_2d, K):
    # Extract rotation and translation from params
    rvec = params[:3]
    tvec = params[3:]
    # Project 3D points to 2D using current R and t
    projected_points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
    projected_points_2d = projected_points_2d.squeeze()

    # Compute the residual (difference between observed and projected points)
    residuals = points_2d - projected_points_2d # N*1
    return residuals.flatten()


def project_3dpoints_to_image(points_3d, w2c_pose, intrinsic_matrix):
    # Step 1: Convert 3D points to homogeneous coordinates (n x 4)
    ones = np.ones((points_3d.shape[0], 1))
    points_3d_h = np.hstack([points_3d, ones])  # (n, 4)
    
    # Step 2: Transform 3D points from world to camera coordinates
    points_cam = (w2c_pose @ points_3d_h.T).T  # Apply world-to-camera transformation (n, 4)
    
    # Keep only the first 3 columns (x, y, z in camera space)
    points_cam = points_cam[:, :3]
    
    # Step 3: Project points to 2D using the camera intrinsic matrix
    points_2d_h = (intrinsic_matrix @ points_cam.T).T  # Apply intrinsic matrix (n, 3)
    
    # Step 4: Normalize to get final 2D pixel coordinates
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2, np.newaxis]
    
    return points_2d

def get_depth_from_3d_points(pts_3d, w2c_pose):
    # Step 1: Convert 3D points to homogeneous coordinates (n x 4)
    num_points = pts_3d.shape[0]
    pts_3d_h = np.hstack([pts_3d, np.ones((num_points, 1))])  # (n, 4)
    
    # Step 2: Transform 3D points from world to camera coordinates using w2c_pose
    pts_camera_h = (w2c_pose @ pts_3d_h.T).T  # (n, 4) points in camera space
    
    # Step 3: Extract the z-coordinates (depth) from the camera space points
    depth = pts_camera_h[:, 2]  # The z-coordinate gives the depth
    
    return depth


def unproject_2d_to_3d(points_2d, c2w_pose, intrinsic_matrix, depths=None):
    # Step 1: Convert 2D points to homogeneous coordinates (u, v, 1)
    num_points = points_2d.shape[0]
    points_2d_h = np.hstack([points_2d, np.ones((num_points, 1))])  # (n, 3)
    
    # Step 2: Backproject to normalized camera coordinates
    intrinsic_inv = np.linalg.inv(intrinsic_matrix)
    points_cam_norm = (intrinsic_inv @ points_2d_h.T).T  # (n, 3)
    
    # Step 3: If depths are not given, assume a default depth (e.g., 1.0)
    if depths is None:
        depths = np.ones(num_points)
    
    # Step 4: Scale by the depth to get points in camera space
    points_cam_3d = points_cam_norm * depths[:, np.newaxis]  # (n, 3)
    
    # Step 5: Convert points to homogeneous coordinates (n x 4)
    points_cam_3d_h = np.hstack([points_cam_3d, np.ones((num_points, 1))])  # (n, 4)
    
    # Step 6: Apply the c2w_pose to convert from camera space to world space
    points_world = (c2w_pose @ points_cam_3d_h.T).T  # (n, 4)
    
    # Step 7: Drop the homogeneous coordinate (last column) to get 3D points
    points_world_3d = points_world[:, :3]
    
    return points_world_3d

def pose_to_rvec_tvec(pose_matrix):
    # Step 1: Extract the 3x3 rotation matrix and the 3x1 translation vector from the 4x4 pose matrix
    R = pose_matrix[:3, :3]  # Rotation matrix (3x3)
    tvec = pose_matrix[:3, 3]  # Translation vector (3x1)
    
    # Step 2: Convert the 3x3 rotation matrix to a rotation vector using Rodrigues' rotation formula
    rvec, _ = cv2.Rodrigues(R)  # rvec will be a 3x1 vector
    
    return rvec.flatten(), tvec.flatten()

def unproject_depth_image(depth_image, intrinsic_matrix, c2w_pose):
    H, W = depth_image.shape
    
    # Step 1: Create the pixel grid (u, v)
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Step 2: Convert pixel coordinates to normalized camera coordinates
    # Intrinsic matrix: K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    
    # Convert u, v to normalized camera coordinates
    x_norm = (u - cx) / fx
    y_norm = (v - cy) / fy
    
    # Step 3: Scale normalized coordinates by the depth image to get 3D camera coordinates
    z_cam = depth_image
    x_cam = x_norm * z_cam
    y_cam = y_norm * z_cam
    
    # Combine x, y, z coordinates into a single array (n, 3)
    points_camera = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)
    
    # Step 4: Convert to homogeneous coordinates for camera-to-world transformation
    points_camera_h = np.hstack([points_camera, np.ones((points_camera.shape[0], 1))])  # (n, 4)
    
    # Step 5: Apply the camera-to-world transformation (c2w_pose)
    points_world_h = (c2w_pose @ points_camera_h.T).T  # (n, 4)
    
    # Step 6: Convert back to 3D by dropping the homogeneous coordinate
    points_world = points_world_h[:, :3]
    
    return points_world

def retrieve_depth_values(depth_image, positions):
    """
    Retrieve the depth values from a depth image at the given positions.

    Parameters:
    - depth_image: A 2D numpy array representing the depth image (H x W)
    - positions: A numpy array of shape (n, 2) where each row is (u, v) pixel coordinates
    
    Returns:
    - A numpy array of depth values at the given positions.
    """
    # Ensure that the positions are valid integers
    positions = np.round(positions).astype(int)
    
    # Separate the (u, v) coordinates into two arrays
    u = positions[:, 0]
    v = positions[:, 1]
    
    # Ensure the indices are within the image bounds
    H, W = depth_image.shape
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)
    
    # Retrieve the depth values at the specified positions
    depth_values = depth_image[v, u]  # Use v for rows (y), u for columns (x)
    
    return depth_values


def make_seq_pairs(imgs, scene_graph='complete'):
    pairs = []
    n = len(imgs)
    if scene_graph == 'complete':  # complete graph
        for i in range(len(imgs)):
            for j in range(i):
                pairs.append((imgs[i], imgs[j]))
    elif scene_graph == 'close4': 
    # Collect indices for the closest 4 images (2 before, 2 after)
        for i in range(n):
            closest_indices = [(i + offset) % n for offset in range(-2, 3) if offset != 0]
            for idx in closest_indices:
                pairs.append((imgs[i], imgs[idx]))
    return pairs

def refine_pose_func(init_pose, high_conf_matched_3d_1, high_conf_matched_2d_2, intrinsics):
    master_rvec, master_tvec = pose_to_rvec_tvec(np.linalg.inv(init_pose))
    high_conf_matched_3d_1 = np.asarray(high_conf_matched_3d_1, dtype=np.float32)
    high_conf_matched_2d_2 = np.asarray(high_conf_matched_2d_2, dtype=np.float32)

    retval, rvec, tvec, inliers = cv2.solvePnPRansac(high_conf_matched_3d_1, high_conf_matched_2d_2, intrinsics, None)
    print("inliers", inliers.shape, "3d points", high_conf_matched_3d_1.shape)
    if retval and inliers is not None:
        inliers = inliers.flatten()
        high_conf_matched_3d_1 = high_conf_matched_3d_1[inliers]
        high_conf_matched_2d_2 = high_conf_matched_2d_2[inliers]
    initial_params = np.hstack((rvec.flatten(), tvec.flatten()))
    result = least_squares(reprojection_error, initial_params, 
                        args=(high_conf_matched_3d_1, high_conf_matched_2d_2, intrinsics),
                        verbose=2,  # Verbose mode to print loss
                        ftol=1e-8, xtol=1e-8)
    optimized_rvec = result.x[:3]
    optimized_tvec = result.x[3:]

    rotation_matrix, _ = cv2.Rodrigues(optimized_rvec)
    opt_w2c = np.eye(4)  # Initialize a 4x4 identity matrix
    opt_w2c[:3, :3] = rotation_matrix  # Set the top-left 3x3 block to the rotation matrix
    opt_w2c[:3, 3] = optimized_tvec  # Set the top-right 3x1 block to the translation vector
    opt_c2w = np.linalg.inv(opt_w2c)
    return opt_c2w, high_conf_matched_3d_1, high_conf_matched_2d_2

def project_point_cloud(K, R, t, P_world):
    # Convert P_world to homogeneous coordinates (add a column of ones)
    P_world_homogeneous = np.hstack((P_world, np.ones((P_world.shape[0], 1))))
    
    # Compute the extrinsic matrix from R and t for easier multiplication
    extrinsic_matrix = np.hstack((R, t.reshape(-1, 1)))
    
    # Transform points to camera coordinate system using extrinsic parameters
    P_camera_homogeneous = np.dot(P_world_homogeneous, extrinsic_matrix.T)
    
    # Project points onto image plane using intrinsic matrix
    P_image_homogeneous = np.dot(P_camera_homogeneous, K.T)
    
    # Convert to Cartesian coordinates by dividing by the z-coordinate
    P_image = P_image_homogeneous[:, :2] / P_image_homogeneous[:, 2].reshape(-1, 1)
    
    return P_image

def find_intersection_and_masks(list_of_3d_points):
    """
    Finds the intersection of a list of N*3 3D points arrays and returns masks indicating the intersection.
    
    :param list_of_3d_points: List of arrays, where each array is N*3 containing 3D points.
    :return: intersection (array), list of masks (one for each input array)
    """
    # Step 1: Convert 3D points to a set of tuples for each array
    sets_of_points = [set(map(tuple, points)) for points in list_of_3d_points]
    
    # Step 2: Find the intersection of all sets
    intersection_set = set.intersection(*sets_of_points)
    
    # Step 3: Convert the intersection set back to a numpy array
    intersection = np.array(list(intersection_set))
    
    # Step 4: Create masks for each input array
    masks = []
    for points in list_of_3d_points:
        mask = np.array([tuple(point) in intersection_set for point in points])
        masks.append(mask)
    
    return intersection, masks


def transform_pc_to_cam_coord(points, c2w):
    """
    Transforms a point cloud from world coordinates to camera coordinates.

    Args:
    points (np.ndarray): Array of points in world coordinates, shape (N, 3).
    c2w (np.ndarray): Camera-to-world transformation matrix, shape (4, 4).

    Returns:
    np.ndarray: Array of points in camera coordinates, shape (N, 3).
    """
    # Calculate the inverse of the camera-to-world matrix (world-to-camera)
    w2c = np.linalg.inv(c2w)

    # Convert points to homogeneous coordinates (add a column of ones)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply the transformation matrix to the points
    points_camera_homogeneous = points_homogeneous.dot(w2c.T)

    # Convert back to non-homogeneous coordinates (discard the last column)
    points_camera = points_camera_homogeneous[:, :3]

    return points_camera


def transform_point_cloud(points_B, transformation_matrix):
    """
    Transforms point cloud B into the coordinate system of A using the given transformation matrix.
    
    :param points_B: Nx3 numpy array representing the point cloud B.
    :param transformation_matrix: 4x4 numpy array representing the transformation from B's coordinate system to A's.
    :return: Nx3 numpy array of transformed points in A's coordinate system.
    """
    # Convert points_B to homogeneous coordinates (Nx4)
    ones = np.ones((points_B.shape[0], 1))
    points_B_hom = np.hstack((points_B, ones))  # Shape: (N, 4)

    # Apply the transformation matrix
    trans_points_B_hom = (transformation_matrix @ points_B_hom.T).T  # Shape: (N, 4)

    # Convert back to 3D coordinates
    trans_points_B = trans_points_B_hom[:, :3]

    return trans_points_B

def compute_scale(P1, P2):
    """
    Computes the scale and translation (bias) between two sets of 3D points.

    :param P1: Nx3 numpy array representing the target set of 3D points.
    :param P2: Nx3 numpy array representing the source set of 3D points.
    :return: scale (s) and translation vector (t) as a tuple.
    """
    # Step 1: Compute the centroids
    centroid_P1 = np.mean(P1, axis=0)
    centroid_P2 = np.mean(P2, axis=0)
    
    # Step 2: Center the points
    P1_centered = P1 - centroid_P1
    P2_centered = P2 - centroid_P2
    
    # Step 3: Compute the scale 
    sigma_source = np.median(np.sqrt(np.sum(P2_centered**2, axis=1)))
    sigma_target = np.median(np.sqrt(np.sum(P1_centered**2, axis=1)))
    scale = sigma_target / sigma_source
    return scale


def master_sparse_recon(image_paths, start_index, end_index, model, known_focal, refine_pose=True, number_match=10000):
    print(f"master on {start_index} and {end_index}, refine pose? {refine_pose}")
    device = 'cuda'
    schedule = 'cosine'
    lr = 0.01
    niter = 300
    
    filelist = [image_paths[start_index], image_paths[end_index]]
    images, ori_size, low_size = load_images(filelist, size=512)
    output = inference([tuple(images)], model, device, batch_size=1, verbose=True)

    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()
    # scene = global_aligner_known_focal(output, device=device, initial_focal=known_focal)

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    if known_focal:
        loss = compute_global_alignment(scene=scene, init="mst", niter=100, schedule='linear', lr=0.01, focal_avg=True, known_focal=known_focal)
    else:
        loss = compute_global_alignment(scene=scene, init="mst", niter=100, schedule='linear', lr=0.01, focal_avg=True)
    
    
    scene = scene.clean_pointcloud()
    intrinsics = to_numpy(scene.get_intrinsics())
    imgs = to_numpy(scene.imgs)
    poses = to_numpy(scene.get_im_poses())
    pts3d = to_numpy(scene.get_pts3d())
    conf = to_numpy(scene.get_conf())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(1.0)))

    pts3d1 = transform_pc_to_cam_coord(pts3d[0].reshape(-1, 3), poses[0]).reshape(low_size[1], low_size[0], 3)
    pts3d2 = transform_pc_to_cam_coord(pts3d[1].reshape(-1, 3), poses[0]).reshape(low_size[1], low_size[0], 3)
    poses = poses @ np.linalg.inv(poses[0]) # the first view as origin


    conf1 = conf[0]
    conf2 = conf[1]

    depth1 = get_depth_from_3d_points(pts3d1.reshape(-1, 3), np.linalg.inv(poses[0]))
    depth1 = depth1.reshape(pts3d1.shape[0], pts3d1.shape[1])

    depth2 = get_depth_from_3d_points(pts3d2.reshape(-1, 3), np.linalg.inv(poses[1]))
    depth2 = depth2.reshape(pts3d2.shape[0], pts3d2.shape[1])

    thres = number_match
    mask1 = conf1 > np.sort(conf1.flatten())[::-1][thres]
    mask2 = conf2 > np.sort(conf2.flatten())[::-1][thres]

    high_conf_xyz = pts3d1[mask1]
    high_conf_xy = project_3dpoints_to_image(high_conf_xyz, np.linalg.inv(poses[0]), intrinsics[0])

    # find 2D-2D matches between the two images
    matches_im0, matches_im1 = fast_reciprocal_NNs(desc1, desc2, subsample_or_initxy1=4,
                                                    device=device, dist='dot', block_size=2**13)
    
    # ignore small border around the edge
    H0, W0 = view1['true_shape'][0]
    valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
        matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

    H1, W1 = view2['true_shape'][0]
    valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
        matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

    valid_matches = valid_matches_im0 & valid_matches_im1
    matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]
    print(matches_im0.shape)

    if refine_pose:
        depth_values = retrieve_depth_values(depth1, matches_im0)
        high_conf_matched_3d_1 = unproject_2d_to_3d(matches_im0, poses[0], intrinsics[0], depth_values)


        high_conf_matched_2d_2 = matches_im1

        # Optimize
        opt_c2w, high_conf_matched_3d_1, high_conf_matched_2d_2 = refine_pose_func(poses[1], high_conf_matched_3d_1, high_conf_matched_2d_2, intrinsics[1])
        intersection, masks = find_intersection_and_masks([matches_im1, high_conf_matched_2d_2])
        updated_matches_im0 = matches_im0[masks[0].squeeze()]
        print("updated inliner matchings", high_conf_matched_3d_1.shape, updated_matches_im0.shape, high_conf_matched_2d_2.shape)
        print("before", end_index, poses[1])
        print("after", end_index, opt_c2w)

        new_pts3d2 = unproject_depth_image(depth2, intrinsics[1], opt_c2w)

        return {"poses": [poses[0], opt_c2w], 
                "intrin":intrinsics,  
                "3d_points": high_conf_matched_3d_1, 
                "matches": [matches_im0, matches_im1],
                "depths": [depth1, depth2],
                "images": [imgs[0], imgs[1]],
                "pc": [pts3d1.reshape(-1, 3), new_pts3d2]
                }
    else:
        return {"poses": [poses[0], opt_c2w], 
                "intrin":intrinsics, 
                "3d_points": None, 
                "matches": [matches_im0, matches_im1],
                "depths": [depth1, depth2],
                "images": [imgs[0], imgs[1]],
                "pc": [pts3d1.reshape(-1, 3), pts3d2.reshape(-1, 3)]
                }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sparse Reconstruction Script")
    parser.add_argument('--model_name', type=str, default="./submodules/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
                        help="Path to the pretrained model")
    parser.add_argument('--source_path', '-s', type=str, required=True, help="Base path to the images")
    parser.add_argument('--n_views', type=int, default=44, help="Number of views to process")
    parser.add_argument('--num_matching', type=int, default=4000, help="Number of matches for sparse reconstruction")
    parser.add_argument('--fragment_size', type=int, default=4, help="Size of each fragment")
    parser.add_argument('--family_sample_rate', type=int, default=2, help="Sample rate for 'Family' scenes")
    parser.add_argument('--default_sample_rate', type=int, default=8, help="Default sample rate for other scenes")
    parser.add_argument('--image_size', type=int, default=512, help="Size of the input images")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for inference")
    parser.add_argument('--output_path', type=str, default=None, help="Path to save the output")

    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(args.source_path, f"sparse/0/sparse_{args.n_views}.pkl")
    else:
        output_dir = os.path.dirname(args.output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    model = AsymmetricMASt3R.from_pretrained(args.model_name).to('cuda')

    print(f"Expect {math.ceil(args.n_views / args.fragment_size)} local gaussians")
    start_time = time.time()

    img_folder_path = os.path.join(args.source_path, "images")
    all_img_list = sorted(os.listdir(img_folder_path))
    sample_rate = args.family_sample_rate if "Family" in img_folder_path else args.default_sample_rate

    ids = np.arange(len(all_img_list))
    i_test = ids[int(sample_rate / 2)::sample_rate]
    i_train = np.array([i for i in ids if i not in i_test])

    test_img_list = [all_img_list[i] for i in i_test]
    tmp_img_list = [all_img_list[i] for i in i_train]
    indices = np.linspace(0, len(tmp_img_list) - 1, args.n_views, dtype=int)
    train_img_list = [tmp_img_list[i] for i in indices]
    image_paths = [os.path.join(img_folder_path, p) for p in train_img_list]
    print(train_img_list)

    images, ori_size, low_size = load_images(image_paths[::args.fragment_size], size=args.image_size)
    pairs = make_seq_pairs(images)
    output = inference(pairs, model, 'cuda', batch_size=args.batch_size, verbose=True)
    
    scene = global_aligner(output, device='cuda', mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = compute_global_alignment(scene=scene, init="mst", niter=300, schedule='linear', lr=0.01, focal_avg=True)
    scene = scene.clean_pointcloud()

    focals = scene.get_focals()
    intrinsics = to_numpy(scene.get_intrinsics())
    known_focal = to_numpy(focals[0])[0]
    global_aligned_poses = to_numpy(scene.get_im_poses())
    pts3d = to_numpy(scene.get_pts3d())
    global_indices = np.arange(len(image_paths))[::args.fragment_size]

    # Initialize storage
    all_poses = []
    all_pair_point_clouds = []
    all_windows = []

    for i in range(1, math.ceil(args.n_views / args.fragment_size) + 1):
        start_idx = args.fragment_size * (i - 1)
        end_idx = min(args.fragment_size * i, args.n_views)
        local_all_corres_dict = []
        local_anchor_matches = []
        local_opt_poses = [np.eye(4)]
        print("local window between ", start_idx, end_idx)
        for jj in range(start_idx+1, end_idx):
            corres_dict = master_sparse_recon(image_paths, start_idx, jj, model, known_focal, number_match=args.num_matching)
            local_all_corres_dict.append(corres_dict)
            local_anchor_matches.append(corres_dict["matches"][0])

        anchor_idx = np.where(global_indices == start_idx)[0][0]
        anchor_3d_all = pts3d[anchor_idx]
        anchor_3d_all_cam = anchor_3d_all.reshape(-1, 3)
        anchor_pose_to_transform = global_aligned_poses[anchor_idx]
        anchor_depth = get_depth_from_3d_points(anchor_3d_all_cam,np.linalg.inv(anchor_pose_to_transform)).reshape(anchor_3d_all.shape[0], anchor_3d_all.shape[1])
        K = local_all_corres_dict[0]["intrin"][0]
        all_poses.append(anchor_pose_to_transform)

       

        local_pc_xyz = [anchor_3d_all_cam]
        local_pc_color = [local_all_corres_dict[0]["images"][0].reshape(-1, 3)]
        cnt=1
        for corres_dict in local_all_corres_dict:
            view0_matchn = corres_dict["matches"][0]
            matchn = corres_dict["matches"][1]
            anchor_3d = anchor_3d_all[view0_matchn[:, 1], view0_matchn[:, 0]]
            
            opt_c2w, anchor_3d_update, new_matchn_update = refine_pose_func(corres_dict["poses"][1], anchor_3d, matchn, corres_dict["intrin"][1])
            local_opt_poses.append(opt_c2w)
            corres_dict["poses"][1] = opt_c2w
            
            anchor_depth_values = retrieve_depth_values(anchor_depth, corres_dict["matches"][0])
            pair_depth_value = retrieve_depth_values(corres_dict["depths"][0], corres_dict["matches"][0])

            anchor_mask = anchor_depth_values < np.percentile(anchor_depth_values, 80)
            pair_mask = pair_depth_value < np.percentile(pair_depth_value, 80)

            matched_anchor_3d = unproject_2d_to_3d(corres_dict["matches"][0][anchor_mask], corres_dict["poses"][0], corres_dict["intrin"][0], depths=anchor_depth_values[anchor_mask])
            matched_pair_3d = unproject_2d_to_3d(corres_dict["matches"][0][pair_mask], corres_dict["poses"][0], corres_dict["intrin"][0], depths=pair_depth_value[pair_mask])
            scale = compute_scale(matched_anchor_3d, matched_pair_3d)
            new_view_2_pc = unproject_depth_image(corres_dict["depths"][1], corres_dict["intrin"][1], opt_c2w)
            view_2_aligned = scale * new_view_2_pc
            print(f"aligning view {start_idx} and view {start_idx+cnt}, scale {scale}")

            if cnt<args.fragment_size:
                local_pc_xyz.append(view_2_aligned)
                local_pc_color.append(corres_dict["images"][1].reshape(-1, 3))
                all_poses.append(opt_c2w)

            corres_dict["3d_points"] = anchor_3d
            corres_dict["depths"][0] = anchor_depth
            corres_dict["pc"][1] = view_2_aligned
            corres_dict["pc"][0] = anchor_3d_all_cam
            cnt+=1

        
        pc_xyz = np.concatenate(local_pc_xyz, axis=0)
        pc_color = np.concatenate(local_pc_color, axis=0)
        all_pair_point_clouds.append(np.concatenate([pc_xyz, pc_color], axis=1))
        all_windows.append(local_all_corres_dict)

    sparse_recon_res = {
        'extrinsics': all_poses,
        'intrinsics': np.tile(intrinsics, (args.n_views, 1, 1)),
        'point_clouds': all_pair_point_clouds,
        'ori_size': ori_size,
        'low_size': low_size,
    }

    with open(args.output_path, 'wb') as file:
        pickle.dump(sparse_recon_res, file)

    print(f"Saved sparse reconstruction results to {args.output_path}")
