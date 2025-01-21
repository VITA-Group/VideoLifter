# This datastructure contains information for local per pair views

import numpy as np
import collections
import struct
from utils.pose_utils import quad2rotation, R_to_quaternion
from icecream import ic

# TODO(Kevin): These codes are repetitive, optmize in future
CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)
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
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)
CAMERA_MODEL_NAMES = dict(
    [(camera_model.model_name, camera_model) for camera_model in CAMERA_MODELS]
)


class Image(BaseImage):
    def qvec2rotmat(self):
        return quad2rotation(self.qvec)


# 2 usage:
# 1)create dummy input for intilization, train_image size , 12
# 2) hold camera infor
class SceneData:
    def __init__(self, index=None, view_size=None, low_size=None):
        self.index = index if index is not None else 1
        self.cameras = {}
        self.images = {}
        self.rgb = None
        self.xyz = None
        self.conf = None
        self.pointcloud = False
        self.size = view_size
        self.depth = None
        self.focal = None
        self.focal_scale = None
        self.low_size = low_size
        self.ori_size = None
        self.view_num_list = None

    def set_cameras(self, ori_size, intrinsics):
        # currenly only support pinhole camera
        model = "PINHOLE"
        self.ori_size = ori_size
        width, height = ori_size
        scale_factor_x = None
        scale_factor_y = None
        K=intrinsics[0]
        scale_factor_x = width / 2 / K[0, 2]
        scale_factor_y = height / 2 / K[1, 2]
        self.focal = K[0, 0]
        self.focal_scale = scale_factor_x
        ic(self.index, self.size + 1, K, self.focal, scale_factor_x, scale_factor_y)
        for i in range(self.index, self.size + 1):
            camera_id = i
            self.cameras[camera_id] = Camera(
                id=camera_id,
                model=model,
                width=width,
                height=height,
                params=[
                    K[0, 0] * scale_factor_x,
                    K[1, 1] * scale_factor_y,
                    width / 2,
                    height / 2,
                ],
            )
    def set_extrinsic(self, poses, train_img_list):
        for i, pose in enumerate(poses, start=self.index):  # Starting index at 1
            pose = np.linalg.inv(pose)
            R = pose[:3, :3]
            t = pose[:3, 3]
            q = R_to_quaternion(R)  # Convert rotation matrix to quaternion
            qvec = q[:4]
            # assume camera_id = image_id
            print("with pose", train_img_list[i - 1])
            self.images[i] = Image(
                id=i,
                qvec=qvec,
                tvec=t.T,
                camera_id=i,
                name=train_img_list[i - 1],
                xys=None,
                point3D_ids=None,
            )
        if self.size is not None:
            for i in range(len(poses)+1, self.size + 1):
                dummy_qvec = np.array([0, 0, 0, 0])
                dummy_tvec = np.array([0, 0, 0])
                self.images[i] = Image(
                    id=i,
                    qvec=dummy_qvec,
                    tvec=dummy_tvec,
                    camera_id=i,
                    name=train_img_list[i - 1],
                    xys=None,
                    point3D_ids=None,
                )

    def set_pt(self, pts_4_3dgs, color_4_3dgs):
        # pts_4_3dgs could be a nx3 point cloud (np.array)or s x i x 3 pointmap(list), would make it more explicit later
        if isinstance(pts_4_3dgs, np.ndarray):
            # input in nx3 numpy point cloud
            self.pointcloud = True
        # else pointcloud = False, --> pointmap
        self.xyz = pts_4_3dgs
        self.rgb = color_4_3dgs

    def set_conf(self, conf):
        self.conf = conf

    def set_view_num_list(self, view_num_list):
        self.view_num_list = view_num_list

    def set_depth(self, depth):
        self.depth = depth
    
    def get_depth(self):
        return self.depth

    def get_transformation(self):
        return self.transformation_matrix

    def get_cameras(self):
        return self.cameras

    def get_extrinsic(self):
        return self.images

    def get_pointcloud(self):
        return self.xyz, self.rgb

    def get_pointcloud_with_index(self, index):
        return self.xyz[index], self.rgb[index]
        
