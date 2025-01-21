
import numpy as np
from scipy.spatial import cKDTree
import cv2
import open3d as o3d
import torch
from utils.graphics_utils import fov2focal
import imageio

def find_closest_points_kdtree(A, B):
    """
    Find the closest points in B for each point in A using a KD-Tree for efficient nearest neighbor search.
    
    Parameters:
        A (numpy.ndarray): Source point cloud of shape (n_points_a, dimensions).
        B (numpy.ndarray): Destination point cloud of shape (n_points_b, dimensions).
    
    Returns:
        numpy.ndarray: An array of shape (n_points_a, dimensions) containing the closest points from B to each point in A.
    """
    tree = cKDTree(B)
    distances, indices = tree.query(A, k=1)
    return B[indices]

def estimate_transformation_scaling(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correcting the rotation matrix for right-hand rule
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    scale = np.sum(np.linalg.norm(BB, axis=1)) / np.sum(np.linalg.norm(AA, axis=1))
    t = centroid_B - scale * R @ centroid_A

    return R, t, scale

def align_point_clouds(source, target, extrin=np.eye(4), max_iterations=100, tolerance=0.001):
    # Initialize the accumulated transformation as identity and zero translation and no scale (scale = 1)
    R_accum = extrin[:3,:3]
    t_accum = extrin[:3,3]
    s_accum = 1.0

    prev_error = float('inf')
    current_source = np.copy(source)

    for i in range(max_iterations):
        # Find correspondences (nearest neighbors)
        closest_points = find_closest_points_kdtree(current_source, target)
        
        # Estimate transformation and scaling
        R, t, s = estimate_transformation_scaling(current_source, closest_points)
        
        # Update the source points with the current estimated transformation
        current_source = s * (R @ current_source.T).T + t
        
        # Accumulate the transformations
        # Update scale
        s_accum *= s
        # Update rotation
        R_accum = R @ R_accum
        # Update translation
        t_accum = s * (R @ t_accum) + t

        # Calculate the mean squared error or a similar metric as a convergence check
        current_error = np.mean(np.linalg.norm(closest_points - current_source, axis=1))
        print(i, current_error)
        
        if abs(prev_error - current_error) < tolerance:
            break
        
        prev_error = current_error

    return R_accum, t_accum, s_accum, current_source

def apply_transformation(points, img, R, t, scale, frame_idx=2, save=False):
    transformed_points = scale * (R @ points.T).T + t
    img = np.array(img)

    # if depth_mask.dtype != np.uint8:
    #     depth_mask = depth_mask.astype(np.uint8)
    # depth_mask = cv2.resize(depth_mask, (img.shape[1], img.shape[0]))
    # depth_mask = depth_mask > 0
    # depth_mask_one = depth_mask.reshape(-1)

    if save:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(transformed_points)
        pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1, 3))
        o3d.io.write_point_cloud(f"frame_{frame_idx}.ply", pcd)

    # return transformed_points[~depth_mask_one], img[~depth_mask].reshape(-1, 3)
    return transformed_points, img.reshape(-1, 3)


def invert_transform(scale, R, t):
    """
    Inverts the transformation defined by the given scale, rotation, and translation.
    
    Args:
    scale (float): Scaling factor applied to the rotation matrix.
    R (numpy.ndarray): Rotation matrix (3x3).
    t (numpy.ndarray): Translation vector (3,).

    Returns:
    numpy.ndarray: Inverted transformation matrix (4x4).
    """
    # Calculate the scaled rotation matrix
    scaled_R_inv = np.linalg.inv(scale * R)
    # Calculate the inverse translation
    inverse_translation = -scaled_R_inv @ t
    # Create the full 4x4 transformation matrix
    T_inv = np.eye(4)
    T_inv[:3, :3] = scaled_R_inv
    T_inv[:3, 3] = inverse_translation
    return T_inv

def update_camera_to_world(original_c2w, scale, R, t):
    """
    Updates the camera-to-world matrix based on the applied transformations.
    
    Args:
    original_c2w (numpy.ndarray): Original camera-to-world matrix (4x4).
    scale (float): Scaling factor applied to the rotation matrix.
    R (numpy.ndarray): Rotation matrix (3x3).
    t (numpy.ndarray): Translation vector (3,).

    Returns:
    numpy.ndarray: Updated camera-to-world matrix (4x4).
    """

    new_c2w = np.eye(4)
    new_c2w[:3,:3]=R
    new_c2w[:3,3]=t

    new_c2w = new_c2w @ original_c2w

    w2c = np.linalg.inv(new_c2w)
    return w2c


def visible_points_from_view(points, c2w, cam, shape):
    """
    Calculate visible points from a camera viewpoint given by c2w matrix and camera intrinsics.
    
    Args:
    points (torch.Tensor): Tensor of shape (N, 3) containing points in world coordinates.
    c2w (torch.Tensor): Camera-to-world transformation matrix of shape (4, 4).
    intrinsics (torch.Tensor): Camera intrinsics matrix of shape (3, 3).
    image_width (int): Width of the camera image.
    image_height (int): Height of the camera image.
    znear (float): Near clipping plane distance.
    zfar (float): Far clipping plane distance.
    
    Returns:
    torch.Tensor: Tensor of visible points in world coordinates.
    """
    image_height, image_width = shape
    znear, zfar = cam.znear, cam.zfar
    fx = fov2focal(cam.FoVx, image_width)
    fy = fov2focal(cam.FoVy, image_height)
    cx = image_width//2
    cy = image_height//2
    # intrinsics = torch.tensor([[fx, 0, ],[0, fy, ],[0,0,1]], dtype=torch.float32, device="cuda")

    w2c = torch.inverse(c2w)

    # Transform point cloud to camera coordinates
    ones = torch.ones((points.shape[0], 1), dtype=torch.float32, device="cuda")
    point_cloud_h = torch.cat((points, ones), dim=1)  # Convert to homogeneous coordinates
    points_cam = point_cloud_h @ w2c.T

    # Project points to image plane
    points_proj = points_cam[:, :3] / points_cam[:, 2:3]
    u = fx * points_proj[:, 0] + cx
    v = fy * points_proj[:, 1] + cy

    # Filter points within image boundaries
    mask = (u >= 0) & (u < image_width) & (v >= 0) & (v < image_height) & (points_cam[:, 2] > 0)
    depth_mask = points_cam[:, 2]<points_cam[:, 2].mean()
    breakpoint()
    return mask & depth_mask
    # return mask
