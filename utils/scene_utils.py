import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "dust3r")))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from utils.local_scene_utils import SceneData


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

def project_visible_points_to_image(points_3d, K, pose):
    """
    Project 3D points to 2D image plane and compute depths.
    
    :param points_3d: Nx3 numpy array of 3D points.
    :param camera_params: Dictionary of camera parameters (focal length, principal point).
    :param R: Rotation matrix (3x3).
    :param t: Translation vector (3x1).
    :return: Dictionary with pixel positions as keys and depths as values.
    """
    R = pose[:3,:3]
    t = pose[:3, 3]
    # Transform points to camera coordinates
    points_cam = R @ points_3d.T + t[:, np.newaxis]  # Broadcasting t for each point
    depths = points_cam[2, :]
    return depths



def get_scene_info_init(
    n_views=28,
    img_base_path="path",
    pair_point_cloud=None,
    ori_size=None,
    low_size=None,
    intrinsics=None,
    extrinsics=None,
    sparse_view_num_list=None
):
    img_folder_path = os.path.join(img_base_path, f"images")
    all_img_list = sorted(os.listdir(img_folder_path))
    ids = np.arange(len(all_img_list))
    if "static_hikes" in img_folder_path:
        sample_rate = 10
        i_test = ids[::sample_rate]
    else:
        sample_rate = 2 if "Family" in img_folder_path else 8
        i_test = ids[int(sample_rate/2)::sample_rate]
    i_train = np.array([i for i in ids if i not in i_test])
    
    train_img_list = sorted(os.listdir(img_folder_path))

    tmp_img_list = [train_img_list[i] for i in i_train]
    indices = np.linspace(0, len(tmp_img_list) - 1, n_views, dtype=int)
    train_img_list = [tmp_img_list[i] for i in indices]

    
    fake_conf = np.zeros_like(pair_point_cloud[:,:3])
    scene_info = SceneData(view_size=n_views, low_size =low_size)
    scene_info.set_cameras(ori_size, intrinsics)
    scene_info.set_extrinsic(extrinsics, train_img_list)
    scene_info.set_pt(pair_point_cloud[:,:3], pair_point_cloud[:,3:])
    scene_info.set_conf(fake_conf)
    scene_info.set_view_num_list(sparse_view_num_list)
    return scene_info
