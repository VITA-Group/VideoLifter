import math
import os
import time
import scipy
import torch
import cv2
import numpy as np
import PIL.Image
from PIL.ImageOps import exif_transpose
from plyfile import PlyData, PlyElement
import torchvision.transforms as tvf
import roma
import re
from pathlib import Path
from typing import List, NamedTuple, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm  # Add this import at the top of your file

import trimesh
from scene.colmap_loader import qvec2rotmat, read_extrinsics_binary, rotmat2qvec
from utils.utils_flow.matching import global_correlation_softmax, local_correlation_softmax
from utils.utils_flow.geometry import coords_grid
from utils.utils_flow.flow_viz import flow_to_image, flow_to_color, save_vis_flow_tofile
import torchvision.transforms.functional as tf



try:
    from pillow_heif import register_heif_opener  # noqa
    register_heif_opener()
    heif_support_enabled = True
except ImportError:
    heif_support_enabled = False
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def save_time(time_dir, process_name, sub_time):
    if isinstance(time_dir, str):
        time_dir = Path(time_dir)
    time_dir.mkdir(parents=True, exist_ok=True)
    minutes, seconds = divmod(sub_time, 60)
    formatted_time = f"{int(minutes)} min {int(seconds)} sec"  
    with open(time_dir / f'train_time.txt', 'a') as f:
        f.write(f'{process_name}: {formatted_time}\n')

def split_train_test_org(image_files, llffhold=8, n_views=None, scene=None):
    print(">> Spliting Train-Test Set: ")
    test_indices = [idx for idx in range(len(image_files)) if idx % llffhold == 0]
    non_test_indices = [idx for idx in range(len(image_files)) if idx % llffhold != 0]
    if n_views is None or n_views == 0:
        n_views = len(non_test_indices)
    sparse_indices = np.linspace(0, len(non_test_indices) - 1, n_views, dtype=int)
    train_indices = [non_test_indices[i] for i in sparse_indices]
    # print(" - sparse_indexs:  ", sparse_indices)
    print(" - train_indices:  ", train_indices)
    print(" - test_indices:   ", test_indices)
    train_img_files = [image_files[i] for i in train_indices]
    test_img_files = [image_files[i] for i in test_indices]

    return train_img_files, test_img_files


def split_train_test(image_files, llffhold=8, n_views=None, scene=None):
    print(">> Spliting Train-Test Set: ")
    ids = np.arange(len(image_files))
    llffhold = 2 if scene=="Family" else 8
    test_indices = ids[int(llffhold/2)::llffhold]
    non_test_indices = np.array([i for i in ids if i not in test_indices])
    # breakpoint()
    if n_views is None or n_views == 0:
        n_views = len(non_test_indices)
    sparse_indices = np.linspace(0, len(non_test_indices) - 1, n_views, dtype=int)
    train_indices = [non_test_indices[i] for i in sparse_indices]
    print(" - sparse_idx:         ", sparse_indices, len(sparse_indices))
    print(" - train_set_indices:  ", train_indices, len(train_indices))
    print(" - test_set_indices:   ", test_indices, len(test_indices))
    train_img_files = [image_files[i] for i in train_indices]
    test_img_files = [image_files[i] for i in test_indices]

    return train_img_files, test_img_files


def get_sorted_image_files(image_dir: str) -> Tuple[List[str], List[str]]:
    """
    Get sorted image files from the given directory.

    Args:
        image_dir (str): Path to the directory containing images.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
            - List of sorted image file paths
            - List of corresponding file suffixes
    """
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.JPG', '.PNG'}
    image_path = Path(image_dir)
    
    def extract_number(filename):
        match = re.search(r'\d+', filename.stem)
        return int(match.group()) if match else float('inf')
    
    image_files = [
        str(f) for f in image_path.iterdir()
        if f.is_file() and f.suffix.lower() in allowed_extensions
    ]
    
    sorted_files = sorted(image_files, key=lambda x: extract_number(Path(x)))
    suffixes = [Path(file).suffix for file in sorted_files]
    
    return sorted_files, suffixes[0]


def rigid_points_registration(pts1, pts2, conf=None):
    R, T, s = roma.rigid_points_registration(
        pts1.reshape(-1, 3), pts2.reshape(-1, 3), weights=conf, compute_scaling=True)
    return s, R, T  # return un-scaled (R, T)


def init_filestructure(save_path, n_views=None):
    if n_views is not None and n_views != 0:        
        sparse_0_path = save_path / f'sparse_{n_views}/0'    
        sparse_1_path = save_path / f'sparse_{n_views}/1'       
        print(f'>> Doing {n_views} views reconstrution!')
    elif n_views is None or n_views == 0:
        sparse_0_path = save_path / 'sparse_0/0'    
        sparse_1_path = save_path / 'sparse_0/1'
        print(f'>> Doing full views reconstrution!')

    save_path.mkdir(exist_ok=True, parents=True)
    sparse_0_path.mkdir(exist_ok=True, parents=True)    
    sparse_1_path.mkdir(exist_ok=True, parents=True)
    return save_path, sparse_0_path, sparse_1_path


import collections
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])
CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model)
                         for camera_model in CAMERA_MODELS])
CAMERA_MODEL_NAMES = dict([(camera_model.model_name, camera_model)
                           for camera_model in CAMERA_MODELS])
      
# Save images and masks
def save_pair_confs_masks(img_files, masks, masks_path, image_suffix):
    for i, (name, mask) in enumerate(zip(img_files, masks)):
        imgname = Path(name).stem
        mask_save_path = masks_path / f"{imgname}{image_suffix}"
        mask = np.repeat(np.expand_dims(mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(mask.astype(np.uint8)).save(mask_save_path)


# Save images and masks
def save_images_and_masks(sparse_0_path, n_views, imgs, global_conf_masks, pair_conf_masks, co_vis_masks, combined_masks, overlapping_masks, image_files, image_suffix):

    images_path       = sparse_0_path / f'imgs_{n_views}'    
    global_conf_masks_path   = sparse_0_path / f'global_conf_masks_{n_views}'
    pair_conf_masks_path = sparse_0_path / f'pair_conf_masks_{n_views}'
    co_vis_masks_path = sparse_0_path / f'co_vis_masks_{n_views}'
    combined_masks_path  = sparse_0_path / f'combined_masks_{n_views}'
    overlapping_masks_path = sparse_0_path / f'overlapping_masks_{n_views}'

    images_path.mkdir(exist_ok=True, parents=True)
    global_conf_masks_path.mkdir(exist_ok=True, parents=True)
    pair_conf_masks_path.mkdir(exist_ok=True, parents=True)
    co_vis_masks_path.mkdir(exist_ok=True, parents=True)
    combined_masks_path.mkdir(exist_ok=True, parents=True)
    overlapping_masks_path.mkdir(exist_ok=True, parents=True)

    for i, (image, name, global_conf_mask, pair_conf_mask, co_vis_mask, combined_mask, overlapping_mask) in enumerate(zip(imgs, image_files, global_conf_masks, pair_conf_masks, co_vis_masks, combined_masks, overlapping_masks)):
        imgname = Path(name).stem
        image_save_path = images_path / f"{imgname}{image_suffix}"
        global_conf_mask_save_path = global_conf_masks_path / f"{imgname}{image_suffix}"
        pair_conf_mask_save_path = pair_conf_masks_path / f"{imgname}{image_suffix}"
        co_vis_mask_save_path = co_vis_masks_path / f"{imgname}{image_suffix}"
        combined_mask_save_path = combined_masks_path / f"{imgname}{image_suffix}"
        overlapping_mask_save_path = overlapping_masks_path / f"{imgname}{image_suffix}"

        # Save overlapping masks
        overlapping_mask = np.repeat(np.expand_dims(overlapping_mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(overlapping_mask.astype(np.uint8)).save(overlapping_mask_save_path)

        # Save images   
        rgb_image = cv2.cvtColor(image * 255, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(image_save_path), rgb_image)

        # Save conf masks
        global_conf_mask = np.repeat(np.expand_dims(global_conf_mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(global_conf_mask.astype(np.uint8)).save(global_conf_mask_save_path)
        pair_conf_mask = np.repeat(np.expand_dims(pair_conf_mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(pair_conf_mask.astype(np.uint8)).save(pair_conf_mask_save_path)

        # Save co-vis masks
        co_vis_mask = np.repeat(np.expand_dims(co_vis_mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(co_vis_mask.astype(np.uint8)).save(co_vis_mask_save_path)

        # Save combined masks
        combined_mask = np.repeat(np.expand_dims(combined_mask, -1), 3, axis=2) * 255
        PIL.Image.fromarray(combined_mask.astype(np.uint8)).save(combined_mask_save_path)


def read_focal_from_cameras_txt(file_path):
    """
    Reads focal lengths from a cameras.txt file where the camera model is 'PINHOLE'.

    Args:
        file_path (str): Path to the cameras.txt file.

    Returns:
        List[float]: A list of focal lengths for 'PINHOLE' camera models.
    """
    focals = []
    with open(file_path, 'r') as file:
        for line in file:
            # Skip comment lines
            if line.startswith('#'):
                continue
            
            # Split the line into parts
            parts = line.strip().split()
            
            # Check if the line has enough parts and the model is 'PINHOLE'
            if len(parts) >= 5 and parts[1] == 'PINHOLE':
                # Extract the focal length (5th element, index 4)
                focal_length = float(parts[4])
                focals.append(focal_length)
    
    return focals


def compute_global_correspondence(feature0, feature1, pred_bidir_flow=False):
    """
    Compute global correspondence between two feature maps.
    
    Args:
        feature0 (torch.Tensor): First feature map of shape [B, C, H, W]
        feature1 (torch.Tensor): Second feature map of shape [B, C, H, W]
    
    Returns:
        torch.Tensor: Correspondence map of shape [B, 2, H, W]
    """
    # Compute flow and probability
    flow, prob = global_correlation_softmax(feature0, feature1, pred_bidir_flow=pred_bidir_flow)
    
    # Get initial grid
    b, _, h, w = feature0.shape
    init_grid = coords_grid(b, h, w).to(feature0.device)  # [B, 2, H, W]
    
    # Compute correspondence
    correspondence = flow + init_grid
    
    return correspondence, flow

def save_flow_visualization(flow, save_path):
    """
    Convert flow to color representation and save as an image.

    Args:
        flow (torch.Tensor): Flow map of shape [B, 2, H, W]
        save_path (str or Path): Path to save the flow visualization
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert flow to numpy and create visualization
    flow_np = flow.cpu().numpy()
    flow_rgb = flow_to_color(flow_np[0].transpose(1, 2, 0))
    
    # Save the flow visualization
    cv2.imwrite(str(save_path), cv2.cvtColor(flow_rgb, cv2.COLOR_RGB2BGR))

def compute_local_correspondence(feature0, feature1, local_radius=4, chunk_size=32):
    """
    Compute local correspondence and flow between two feature maps using a sliding window approach.
    
    Args:
        feature0 (torch.Tensor): First feature map of shape [B, C, H, W]
        feature1 (torch.Tensor): Second feature map of shape [B, C, H, W]
        local_radius (int): Radius for local correlation window
        chunk_size (int): Number of rows to process at once to save memory
    
    Returns:
        tuple: (correspondence, flow)
            - correspondence (torch.Tensor): Correspondence map of shape [B, 2, H, W]
            - flow (torch.Tensor): Flow map of shape [B, 2, H, W]
    """
    b, c, h, w = feature0.shape
    device = feature0.device
    
    # Initialize the output correspondence and flow maps
    correspondence = torch.zeros(b, 2, h, w, device=device)
    flow = torch.zeros(b, 2, h, w, device=device)
    
    # Process the image in chunks to save memory
    for i in range(0, h, chunk_size):
        end = min(i + chunk_size, h)
        
        # Extract chunks from both feature maps
        chunk0 = feature0[:, :, i:end, :]
        chunk1 = feature1[:, :, max(0, i-local_radius):min(h, end+local_radius), :]
        
        # Compute local correlation for the chunk
        flow_chunk, _ = local_correlation_softmax(chunk0, chunk1, local_radius)
        
        # Convert flow to correspondence
        init_grid_chunk = coords_grid(b, end-i, w, device=device)
        correspondence_chunk = flow_chunk + init_grid_chunk
        
        # Update the correspondence and flow maps
        correspondence[:, :, i:end, :] = correspondence_chunk
        flow[:, :, i:end, :] = flow_chunk
    
    return correspondence, flow

# You can add any additional utility functions here if needed


def save_feature_visualization(feature, save_path, title='Feature Visualization'):
    """
    Save the visualization of feature maps to a file.

    Parameters:
    - feature: torch.Tensor, the feature map to visualize, expected shape (C, H, W)
    - save_path: str, the path to save the visualization image
    - title: str, the title for the visualization
    """
    feature_np = feature.cpu().numpy()  # Convert to numpy array
    num_channels = feature_np.shape[0]
    plt.figure(figsize=(num_channels * 3, 3))
    for i in range(num_channels):
        plt.subplot(1, num_channels, i + 1)
        plt.imshow(feature_np[i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(title)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def save_merged_feature_visualization(feature, save_path, title='Merged Feature Visualization', method='mean'):
    """
    Save the merged visualization of feature maps to a file.

    Parameters:
    - feature: torch.Tensor, the feature map to visualize, expected shape (C, H, W)
    - save_path: str, the path to save the visualization image
    - title: str, the title for the visualization
    - method: str, the method to merge channels ('mean' or 'max')
    """
    feature_np = feature.cpu().numpy()  # Convert to numpy array

    if method == 'mean':
        merged_feature = np.mean(feature_np, axis=0)
    elif method == 'max':
        merged_feature = np.max(feature_np, axis=0)
    else:
        raise ValueError("Method must be 'mean' or 'max'.")

    plt.figure(figsize=(5, 5))
    plt.imshow(merged_feature, cmap='viridis')
    plt.axis('off')
    plt.title(title)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save the figure
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def feature_save(tensor,path_name,name):
    # tensor = torchvision.utils.make_grid(tensor.transpose(1,0))
    tensor = torch.mean(tensor,dim=1)
    inp = tensor.detach().cpu().numpy().transpose(1,2,0)
    inp = inp.squeeze(2)
    inp = (inp - np.min(inp)) / (np.max(inp) - np.min(inp))
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    inp = cv2.applyColorMap(np.uint8(inp * 255.0),cv2.COLORMAP_JET)
    cv2.imwrite(path_name  + name, inp)


def cal_co_vis_mask(points, depths, curr_depth_map, depth_threshold, camera_intrinsics, extrinsics_w2c):

    h, w = curr_depth_map.shape
    overlapping_mask = np.zeros((h, w), dtype=bool)
    # Project 3D points to image j
    points_2d, _ = project_points(points, camera_intrinsics, extrinsics_w2c)
    
    # Check if points are within image bounds
    valid_points = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                   (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        
    # Check depth consistency using vectorized operations
    valid_points_2d = points_2d[valid_points].astype(int)
    valid_depths = depths[valid_points]

    # Extract x and y coordinates
    x_coords, y_coords = valid_points_2d[:, 0], valid_points_2d[:, 1]

    # Compute depth differences
    depth_differences = np.abs(valid_depths - curr_depth_map[y_coords, x_coords])

    # Create a mask for points where the depth difference is below the threshold
    consistent_depth_mask = depth_differences < depth_threshold

    # Update the overlapping masks using the consistent depth mask
    overlapping_mask[y_coords[consistent_depth_mask], x_coords[consistent_depth_mask]] = True

    return overlapping_mask

def normalize_depth(depth_map):
    """Normalize the depth map to a range between 0 and 1."""
    return (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

def compute_overlapping_mask_2(sorted_conf_indices, depthmaps, pointmaps, camera_intrinsics, extrinsics_w2c, image_sizes, depth_threshold=0.1):

    num_images, h, w, _ = image_sizes
    pointmaps = pointmaps.reshape(num_images, h, w, 3)
    overlapping_masks = np.zeros((num_images, h, w), dtype=bool)
    
    for i, curr_map_idx in tqdm(enumerate(sorted_conf_indices), total=len(sorted_conf_indices)):

        if i == 0:
            continue

        idx_before = sorted_conf_indices[:i]

        # get partial pointmaps and depthmaps
        points_before = pointmaps[idx_before].reshape(-1, 3)
        depths_before = depthmaps[idx_before].reshape(-1)    
        curr_depth_map = depthmaps[curr_map_idx].reshape(h, w)

        # normalize depth for comparison
        depths_before = normalize_depth(depths_before)
        curr_depth_map = normalize_depth(curr_depth_map)
        

        before_mask = cal_co_vis_mask(points_before, depths_before, curr_depth_map, depth_threshold, camera_intrinsics[curr_map_idx], extrinsics_w2c[curr_map_idx])
        overlapping_masks[curr_map_idx] = before_mask
        
    return overlapping_masks

def compute_overlapping_mask(depthmaps, pointmaps, camera_intrinsics, extrinsics_w2c, image_sizes, depth_threshold=0.1):
    num_images, h, w, _ = image_sizes
    pointmaps = pointmaps.reshape(num_images, h, w, 3)
    overlapping_masks = np.zeros((num_images, h, w), dtype=bool)
    
    for i in range(num_images):
        # Exclude the current pointmap
        points_3d = pointmaps[np.arange(num_images) != i].reshape(-1, 3)        
        depths = depthmaps[np.arange(num_images) != i].reshape(-1)
        depth_map_i = depthmaps[i].reshape(h, w)

        # normalize depth for comparison
        depths = normalize_depth(depths)
        depth_map_i = normalize_depth(depth_map_i)
        
        for j in range(num_images):
            if i == j:
                continue
            
            # Project 3D points to image j
            points_2d, _ = project_points(points_3d, camera_intrinsics[j], extrinsics_w2c[i])
            
            # Check if points are within image bounds
            valid_points = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                           (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
            
            # Check depth consistency using vectorized operations
            valid_points_2d = points_2d[valid_points].astype(int)
            valid_depths = depths[valid_points]

            # Extract x and y coordinates
            x_coords, y_coords = valid_points_2d[:, 0], valid_points_2d[:, 1]

            # Compute depth differences
            depth_differences = np.abs(valid_depths - depth_map_i[y_coords, x_coords])

            # Create a mask for points where the depth difference is below the threshold
            consistent_depth_mask = depth_differences < depth_threshold

            # Update the overlapping masks using the consistent depth mask
            overlapping_masks[i][y_coords[consistent_depth_mask], x_coords[consistent_depth_mask]] = True
    return overlapping_masks



def project_points(points_3d, intrinsics, extrinsics):
    # Convert to homogeneous coordinates
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # Apply extrinsic matrix
    points_camera = np.dot(extrinsics, points_3d_homogeneous.T).T
    
    # Apply intrinsic matrix
    points_2d_homogeneous = np.dot(intrinsics, points_camera[:, :3].T).T
    
    # Convert to 2D coordinates
    points_2d = points_2d_homogeneous[:, :2] / points_2d_homogeneous[:, 2:]
    depths = points_camera[:, 2]
    
    return points_2d, depths

def read_colmap_gt_pose(gt_pose_path, llffhold=8):
    # colmap_cam_extrinsics = read_extrinsics_binary(gt_pose_path + '/triangulated/images.bin')
    colmap_cam_extrinsics = read_extrinsics_binary(gt_pose_path + '/sparse/0/images.bin')
    all_pose=[]
    print("Loading colmap gt train pose:")
    for idx, key in enumerate(colmap_cam_extrinsics):
        # if idx % llffhold == 0:
        extr = colmap_cam_extrinsics[key]
        # print(idx, extr.name)
        # R = np.transpose(qvec2rotmat(extr.qvec))
        R = np.array(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        pose = np.eye(4,4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        all_pose.append(pose)
    colmap_pose = np.array(all_pose)
    return colmap_pose

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = PIL.Image.open(renders_dir / fname)
        gt = PIL.Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def align_pose(pose1, pose2):
    mtx1 = np.array(pose1, dtype=np.double, copy=True)
    mtx2 = np.array(pose2, dtype=np.double, copy=True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)

    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
    mtx2 = mtx2 * s
    print("scale", s)

    return mtx1, mtx2, R

