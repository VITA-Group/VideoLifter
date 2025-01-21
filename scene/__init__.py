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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from icecream import ic
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, getProjectionMatrix, \
    compute_scale_gaussian_by_project_scene_info, compute_scale_gaussian_by_project_pcd
import torch

class Scene:

    # image folder would have al train,test images
    gaussians: GaussianModel

    # scene_data:contains information for intrinsic, extrinsic and pointcloud
    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration=None,
        opt=None,
        shuffle=True,
        resolution_scales=[1.0],
        scene_data=None,
        progressive = True,
        progressive_index = 2
    ):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.progressive = progressive
        self.progressive_index = progressive_index
        # The following

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        # FIXME: in later version don't save scene_data to prevent memory explode
        self.scene_data = scene_data
        ic(args.source_path)
        ic(scene_data)
        if scene_data is not None and scene_data.pointcloud is False:
            ic("Directly load pointmap")
            scene_info = sceneLoadTypeCallbacks["Progressive_pt"](
                args.source_path, args.images, args.eval, scene_data, args, opt
            )
        # TODO !!!find a flag to add!!!
        elif scene_data is not None and scene_data.pointcloud:
            ic("Directly load pointclouds from pointcloud")
            scene_info = sceneLoadTypeCallbacks["Progressive"](
                args.source_path, args.images, args.eval, scene_data, args, opt
            )
        # for each point cloud saved
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            # FIXME (maybe require a fix here?)
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, args, opt
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print(len(self.train_cameras[resolution_scale]))
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if self.loaded_iter:
            self.gaussians.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        elif self.progressive:
            scale_gaussian = compute_scale_gaussian_by_project_scene_info(scene_info, scene_data.view_num_list)
            self.gaussians.create_from_pcd(scene_info.point_cloud, scene_data.conf, scale_gaussian, self.cameras_extent)
            self.gaussians.init_RT_seq(self.train_cameras, self.progressive_index)
        ic(self.gaussians._xyz.shape)

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))


    def save_list(self, iteration, gaussian_list): # for localrf dataset
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        for ii, local_gaussian in enumerate(gaussian_list):
            local_gaussian.save_ply(os.path.join(point_cloud_path, f"point_cloud_{ii}.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale][: self.progressive_index]
    
    def getAllTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
    
    def addTrainViewPose_progressive(self, intrin, extrin, scale=1.0):
        ic("new view pose in")
        index = min(self.progressive_index, len(self.train_cameras[scale]))
        ic(index)
        cam = self.train_cameras[scale][index]
        
        cam.R = extrin[:3, :3].T
        cam.T = extrin[:3, 3]
        cam.world_view_transform = torch.tensor(getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)).transpose(0, 1).cuda()
        cam.projection_matrix = getProjectionMatrix(znear=cam.znear, zfar=cam.zfar, fovX=cam.FoVx, fovY=cam.FoVy).transpose(0,1).cuda()
        cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        self.train_cameras[scale][index] = cam
       
        self.progressive_index += 1
        return cam

    def addTrainView_progressive(self, intrin, extrin, pts, colors, scale=1.0):
        ic("new view in")
        self.gaussians.add_points_from_xyz_rgb(pts, colors, None)
        
    def getTrainCameras_progressive(self, scale=1.0):
        ic("after adding new view")
        index = min(self.progressive_index, len(self.train_cameras[scale]))
        result = self.train_cameras[scale][:index]
        ic(self.gaussians._xyz.shape)
        return result

