#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.local_scene_utils import SceneData
from icecream import ic


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask: np.array

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    train_poses: list
    test_poses: list

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal
    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras_wTrainFocal(cam_extrinsics, cam_intrinsics, images_folder, eval, model_path):
    # TODO: Also add a modify train_camera_info function.
    train_cam_json = os.path.join(model_path, "cameras.json")
    with open(train_cam_json, 'r') as file:
        data = json.load(file)
    train_fx = data[0]['fx']
    train_fy = data[0]['fy']

    cam_infos = []
    poses=[]
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        if eval:
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[1]
            uid = idx+1
        else:
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            uid = intr.id

        # TODO: //2 needed for co3d 429
        height = intr.height
        width = intr.width
     
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        pose =  np.vstack((np.hstack((R, T.reshape(3,-1))),np.array([[0, 0, 0, 1]])))
        poses.append(pose)
        
        focal_length_x = train_fx 
        focal_length_y = train_fy
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        print(intr.model, intr.params, focal_length_x)

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        print(image_path)
        image = Image.open(image_path)
        

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, mask=None)

        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos, poses



def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, eval):
    cam_infos = []
    poses=[]
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        if eval:
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[1]
            uid = idx+1
        else:
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            uid = intr.id

        height = intr.height
        width = intr.width            
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        pose =  np.vstack((np.hstack((R, T.reshape(3,-1))),np.array([[0, 0, 0, 1]])))
        poses.append(pose)
        
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        print(intr.model, intr.params, focal_length_x)
        # if eval:
        #     tmp = os.path.dirname(os.path.dirname(os.path.join(images_folder)))
        #     if os.path.exists( os.path.join(tmp, 'images_8')):
        #         all_images_folder = os.path.join(tmp, 'images_8')
        #     else:
        #         all_images_folder = os.path.join(tmp, 'images')
        #     image_path = os.path.join(all_images_folder, os.path.basename(extr.name))
        # else:
        # wenyan: change to directly reading from images folder
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        print(image_path)
        image = Image.open(image_path)

        mask_path = image_path.replace("images", "masks")
        if os.path.exists(mask_path):
            print(mask_path)
            mask = Image.open(mask_path)
        else:
            mask=None
        

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, mask=mask)

        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos, poses

# For interpolated video, open when only render interpolated video
def readColmapCamerasInterp(cam_extrinsics, cam_intrinsics, images_folder, model_path):
    
    pose_interpolated_path = model_path + 'pose/pose_interpolated.npy'
    pose_interpolated = np.load(pose_interpolated_path)
    intr = cam_intrinsics[1]

    train_cam_json = os.path.join(model_path, "cameras.json")
    with open(train_cam_json, 'r') as file:
        data = json.load(file)
    train_fx = data[0]['fx']
    train_fy = data[0]['fy']

    cam_infos = []
    poses=[]
    for idx, pose_npy in enumerate(pose_interpolated):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, pose_interpolated.shape[0]))
        sys.stdout.flush()

        extr = pose_npy
        intr = intr
        height = intr.height
        width = intr.width

        uid = idx
        R = extr[:3, :3].transpose()
        T = extr[:3, 3]
        pose =  np.vstack((np.hstack((R, T.reshape(3,-1))),np.array([[0, 0, 0, 1]])))
        poses.append(pose)
        focal_length_x = train_fx 
        focal_length_y = train_fy 
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)


        images_list = os.listdir(os.path.join(images_folder))
        # Dummy image here 
        image_name_0 = images_list[0]
        image_name = str(idx).zfill(4)
        image = Image.open(images_folder + '/' + image_name_0)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=images_folder, image_name=image_name, width=width, height=height, mask=None)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, poses


def readColmapCamerasInterp_CO3D(cam_extrinsics, cam_intrinsics, images_folder, model_path):
    
    pose_interpolated_path = model_path + 'pose/pose_interpolated.npy'
    pose_interpolated = np.load(pose_interpolated_path)
    intr = cam_intrinsics[1]

    train_cam_json = os.path.join(model_path, "cameras.json")
    with open(train_cam_json, 'r') as file:
        data = json.load(file)
    train_fx = data[0]['fx']
    train_fy = data[0]['fy']

    cam_infos = []
    poses=[]
    for idx, pose_npy in enumerate(pose_interpolated):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, pose_interpolated.shape[0]))
        sys.stdout.flush()

        extr = pose_npy
        intr = intr

        # TODO: //2 for co3d 429 only
        height = intr.height
        width = intr.width

        uid = idx
        R = extr[:3, :3].transpose()
        T = extr[:3, 3]
        pose =  np.vstack((np.hstack((R, T.reshape(3,-1))),np.array([[0, 0, 0, 1]])))
        poses.append(pose)
        print(intr.model, intr.params)
        focal_length_x = train_fx
        focal_length_y = train_fy
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        images_list = os.listdir(os.path.join(images_folder))
        # Dummy image here 
        image_name_0 = images_list[0]
        image_name = str(idx).zfill(4)
        image = Image.open(images_folder + '/' + image_name_0)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=images_folder, image_name=image_name, width=width, height=height, mask=None)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos, poses

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, args, opt, llffhold=2):
    cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)


    ic(cameras_extrinsic_file, cameras_intrinsic_file)

    reading_dir = "images" if images == None else images

    if opt.get_video:
        if "co3d" in path:
            cam_infos_unsorted, poses = readColmapCamerasInterp_CO3D(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), model_path=args.model_path)
        else:
            cam_infos_unsorted, poses = readColmapCamerasInterp(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), model_path=args.model_path)
    else:
        cam_infos_unsorted, poses = readColmapCameras_wTrainFocal(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), eval=eval, model_path=args.model_path)
    sorting_indices = sorted(range(len(cam_infos_unsorted)), key=lambda x: cam_infos_unsorted[x].image_name)
    cam_infos = [cam_infos_unsorted[i] for i in sorting_indices]
    sorted_poses = [poses[i] for i in sorting_indices]
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        ids = np.arange(len(cam_infos))
        sample_rate = 2 if "Family" in path else 8
        i_test = ids[int(sample_rate/2)::sample_rate]

        # uniform sample
        i_train = np.array([i for i in ids if i not in i_test])
        indices = np.linspace(0, len(i_train) - 1, opt.n_views, dtype=int)
        i_train = [i_train[i] for i in indices]
        
        train_cam_infos = [cam_infos[i] for i in i_train]
        test_cam_infos = [cam_infos[i] for i in i_test]
        train_poses = [sorted_poses[i] for i in i_train]
        test_poses = [sorted_poses[i] for i in i_test]
        

    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        train_poses = sorted_poses
        test_poses = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           train_poses=train_poses,
                           test_poses=test_poses)
    return scene_info


def readDust3RInfo(path, images, eval, scene_data: SceneData, args, opt, llffhold=2):
    cam_extrinsics = scene_data.get_extrinsic()
    cam_intrinsics = scene_data.get_cameras()
    reading_dir = "images" if images == None else images
    if opt.get_video:
        cam_infos_unsorted, poses = readColmapCamerasInterp(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(path, reading_dir),
            model_path=args.model_path,
        )
    else:
        cam_infos_unsorted, poses = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(path, reading_dir),
            eval=eval,
        )
    cam_infos = cam_infos_unsorted
    sorted_poses = poses
    

    if eval:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos
        train_poses = sorted_poses
        test_poses = sorted_poses
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
        train_poses = sorted_poses
        test_poses = []

    nerf_normalization = getNerfppNorm(train_cam_infos)
    try:
        ic("in get point cloud")
        xyz, rgb = scene_data.get_pointcloud()
        pcd = BasicPointCloud(points=xyz, colors=rgb, normals=None)
        ic("------------------successfuly created pcd---------------")
    except:
        pcd = None
        ic("random pointcloud")

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None,
        train_poses=train_poses,
        test_poses=test_poses,
    )
    return scene_info


def readDust3RInfo_pt(path, images, eval, scene_data: SceneData, args, opt, llffhold=2):
    ic("in readDust3RInfo_pt")
    cam_extrinsics = scene_data.get_extrinsic()
    cam_intrinsics = scene_data.get_cameras()

    reading_dir = "images" if images == None else images

    if opt.get_video:
        cam_infos_unsorted, poses = readColmapCamerasInterp(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(path, reading_dir),
            model_path=args.model_path,
        )
    else:
        cam_infos_unsorted, poses = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(path, reading_dir),
            eval=eval,
        )
    sorting_indices = sorted(
        range(len(cam_infos_unsorted)), key=lambda x: cam_infos_unsorted[x].image_name
    )
    cam_infos = [cam_infos_unsorted[i] for i in sorting_indices]
    sorted_poses = [poses[i] for i in sorting_indices]
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos
        train_poses = sorted_poses
        test_poses = sorted_poses

    else:
        train_cam_infos = cam_infos
        test_cam_infos = cam_infos
        train_poses = sorted_poses
        test_poses = sorted_poses

    nerf_normalization = getNerfppNorm(train_cam_infos)
    # try:
    xyz, rgb = scene_data.get_pointcloud_with_index(1)
    ic(xyz.shape)
    xyz2, rgb2 = scene_data.get_pointcloud_with_index(1)
    xyz = np.concatenate([xyz, xyz2])
    rgb = np.concatenate([rgb, rgb2])
    ic(xyz.shape)
    ic(rgb.shape)
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=None)
    print("------------------successfuly created pcd----------------")


    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=None,
        train_poses=train_poses,
        test_poses=test_poses,
    )
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1], mask=None))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Progressive": readDust3RInfo,
    "Progressive_pt": readDust3RInfo_pt,
}
