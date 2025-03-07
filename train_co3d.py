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
import numpy as np
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_depth_loss
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2_torch, getWorld2View2, compute_scale_gaussian_by_project_pair_pcd
from utils.pose_utils import get_camera_from_tensor, quadmultiply, get_tensor_from_camera
from utils.camera_utils import generate_interpolated_path
from utils.camera_utils import visualizer
from utils.logger_utils import prepare_output_and_logger, training_report
from utils.icp_utils import align_point_clouds, apply_transformation, visible_points_from_view
from utils.general_utils import inverse_sigmoid
from utils.scene_utils import get_scene_info_init
import torchvision
from icecream import ic
import cv2
import random
import math
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import pickle

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

### for high_resolution
def save_pose(path, quat_pose, train_cams):
    output_poses=[]
    # # breakpoint()
    # pose_before = np.load('output/Horse_hr_3_view/pose_before.npy')
    index_colmap = [cam.colmap_id//1 for cam in train_cams]
    # index_colmap = [cam.colmap_id//2 for cam in train_cams]
    for quat_t in quat_pose:
        c2w = get_camera_from_tensor(quat_t)
        output_poses.append(c2w)
    colmap_poses = []
    for i in range(len(index_colmap)):
        # ind = i
        ind = index_colmap.index(i+1)
        bb=output_poses[ind]
        bb = bb#.inverse()
        colmap_poses.append(bb)
    colmap_poses = torch.stack(colmap_poses).detach().cpu().numpy()
    np.save(path, colmap_poses)


def initialize_camera_pose(prev_pose1, prev_pose2):
    with torch.no_grad():
        # Initialize the camera pose for the current frame based on a constant velocity model
        # Rotation
        prev_rot1 = F.normalize(prev_pose1[:4].unsqueeze(0).detach())
        prev_rot2 = F.normalize(prev_pose2[:4].unsqueeze(0).detach())
        new_rot = F.normalize(prev_rot1 + (prev_rot1 - prev_rot2)).squeeze(0)
        # Translation
        prev_tran1 = prev_pose1[4:].detach()
        prev_tran2 = prev_pose2[4:].detach()
        new_tran = prev_tran1 + (prev_tran1 - prev_tran2)
        new_pose = torch.cat((new_rot, new_tran))
    return new_pose

def custom_random_select(left, right):
    if random.random() < 0.5:  # 50% probability
        if right - 2> left:
            return random.randint(left, right - 2)
        else:
            return left
    else:
        return left - 1

def random_sample(left, right):
    # Calculate the lambda parameter for the exponential distribution
    # Lambda controls the rate at which the exponential distribution decays
    decay_rate = 2.0  # Adjust this parameter based on your preference
    if right <= left:
        return left
    lam = 1 / (right - left) * decay_rate

    # Generate a random sample using the exponential distribution
    # The closer the value to right, the higher the probability
    sample = np.random.exponential(scale=1/lam) + left

    # Ensure the sampled value is within the specified range
    sample = min(sample, right)  # Cap the value to right if it exceeds right
    sample = max(sample, left)   # Ensure the value is not less than left

    return sample

def save_interpolate_pose(model_path, iter, n_views):
    org_pose = np.load(model_path + f"/pose_{iter}.npy")
    visualizer(org_pose, ["green" for _ in org_pose], model_path + "/poses_org.png")

    n_interp = int(10 * 30 / n_views)  # 10second, fps=30
    all_inter_pose = []
    for i in range(n_views-1):
        tmp_inter_pose = generate_interpolated_path(poses=org_pose[i:i+2], n_interp=n_interp)
        all_inter_pose.append(tmp_inter_pose)
    all_inter_pose = np.array(all_inter_pose).reshape(-1, 3, 4)
    inter_pose_list = []

    for p in all_inter_pose:
        tmp_view = np.eye(4)
        # tmp_view[:3] = p
        tmp_view[:3, :3] = p[:3, :3]
        tmp_view[:3, 3] = p[:3, 3]
        inter_pose_list.append(tmp_view)
    inter_pose = np.stack(inter_pose_list, 0)
    visualizer(inter_pose, ["blue" for _ in inter_pose], model_path + "/poses_interpolated.png")
    np.save(model_path + "/pose_interpolated.npy", inter_pose)

def optimize_pose(gaussians, view_cam, pipe, bg, pose_candidate, out_img_folder=""):
    num_iters_pose_opt = 200
    patience = 10      # Number of iterations to wait for improvement
    min_delta = 1e-4   # Minimum change in loss to qualify as an improvement

    # Initialize pose parameters with gradients
    camera_tensor_T = pose_candidate[-3:].detach().clone().requires_grad_()
    camera_tensor_q = pose_candidate[:4].detach().clone().requires_grad_()

    # Setup optimizer
    pose_optimizer = torch.optim.Adam(
        [
            {"params": [camera_tensor_T], "lr": 0.0001, "name": "trans"},
            {"params": [camera_tensor_q], "lr": 0.0001, "name": "rot"},
        ]
    )

    # Prepare for optimization
    pose_optimizer.zero_grad(set_to_none=True)
    progress_bar_pose = tqdm(range(num_iters_pose_opt), desc=f"Optimizing Pose")
    candidate_q = camera_tensor_q.clone().detach()
    candidate_T = camera_tensor_T.clone().detach()
    current_min_loss = float('inf')
    image_gt = view_cam.original_image.cuda()
    no_improve_counter = 0

    for iter in range(num_iters_pose_opt):
        # Render the image with current pose
        rendering = render(
            view_cam, gaussians, pipe, bg,
            camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]).unsqueeze(0)
        )
        image = rendering["render"]
        depth = rendering["depth"]

        # Create mask and compute loss
        mask = (depth > 0).bool().expand_as(image_gt)
        loss = torch.abs(image_gt[mask] - image[mask]).mean()

        # Save the best pose parameters if loss improves
        if loss.item() < current_min_loss - min_delta:
            current_min_loss = loss.item()
            candidate_q = camera_tensor_q.clone().detach()
            candidate_T = camera_tensor_T.clone().detach()
            no_improve_counter = 0  # Reset counter
        else:
            no_improve_counter += 1  # Increment counter

        # Early stopping condition
        if no_improve_counter >= patience:
            print(f"Early stopping at iteration {iter} due to no significant improvement.")
            break

        # Backpropagation and optimizer step
        loss.backward()
        if iter % 10 == 0:
            print(f"Iteration {iter}, Loss: {loss.item():.6f}")
        pose_optimizer.step()
        pose_optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

    # Return the optimized pose
    opt_pose = torch.cat([candidate_q, candidate_T])
    return opt_pose


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def gaussian_training(gaussians, scene, start_index, end_index, pipe, bg, iterations, opt, flag_global=False, sh_up=False, out_img_folder=""):
    viewpoint_stack=None
    
    # add lr schedular
    current_level = int(math.log2(end_index-start_index))
    optimized_iters = 0
    for past_level in range(0, current_level):
        optimized_iters += past_level*200
    print(start_index, end_index, current_level, optimized_iters)
    for iteration in range(0, iterations):
        if optimized_iters > 0:
            level_iter = optimized_iters+iteration
            gaussians.update_learning_rate(level_iter)
        if sh_up:
            if iteration == 200:
                print("####### increase sh degree #######")
                gaussians.oneupSHdegree()

        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()[start_index: end_index]
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        pose = gaussians.get_RT(viewpoint_cam.uid-start_index)
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, camera_pose=pose)
        image = render_pkg["render"]
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = ((1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)))
        if iteration%10==0:
            print(iteration, viewpoint_cam.uid, viewpoint_cam.uid-start_index, loss.item())
        loss.backward()
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none = True)

    return gaussians, scene

def save_gaussian(gaussian, model_path, iteration):
    point_cloud_path = os.path.join(
            model_path, "point_cloud/iteration_{}".format(iteration)
        )
    gaussian.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

# @profile
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    sparse_recon_res_path = os.path.join(args.model_path, f"sparse/0/sparse_{args.n_views}.pkl")
    with open(sparse_recon_res_path, 'rb') as file:
        sparse_recon_res = pickle.load(file)
    sparse_low_size = sparse_recon_res["low_size"]
    sparse_ori_size = sparse_recon_res["ori_size"]
    sparse_intrinsics = sparse_recon_res["intrinsics"]
    sparse_extrinsics = sparse_recon_res["extrinsics"]
    sparse_all_pair_point_clous = sparse_recon_res["point_clouds"]
    
    scene_info = get_scene_info_init(
                img_base_path=args.sparse_image_folder, n_views=args.n_views, pair_point_cloud=sparse_all_pair_point_clous[0],
                ori_size=sparse_ori_size, low_size=sparse_low_size, intrinsics=sparse_intrinsics[0:4], extrinsics=sparse_extrinsics)

    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(
        dataset, gaussians, opt=args, shuffle=False, scene_data=scene_info, progressive_index=4
    )
    gaussians.training_setup(opt)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    base_num_iters_gaussian_opt = 400
    alpha_threshold = 0.90
    global_iteration = 0
    prev_level_list=[]
    cur_level_list=[]
    global_iter = 0
    bg = torch.rand((3), device="cuda") if opt.random_background else background
    loww, lowh = scene_info.low_size
    oriw, orih = scene_info.ori_size
    w_scaler = oriw/loww
    h_scaler = orih/lowh
    for level in range(2, math.ceil(math.log2(args.n_views))+1):
        print(level)
        if level==2:
            for count in range(max(math.ceil(args.n_views/4), 1)):
                if count==0:
                    start_index = count* 4
                    end_index = min((count+1) * 4, args.n_views)
                    gaussians, scene = gaussian_training(gaussians, scene, start_index, end_index, pipe, bg, base_num_iters_gaussian_opt, opt, sh_up=True, out_img_folder=scene.model_path)
                    global_iter += base_num_iters_gaussian_opt
                else:
                    local_gaussians = GaussianModel(dataset.sh_degree)
                    start_index = count* 4
                    end_index = min((count+1) *4, args.n_views)
                    naive_pair_poses= sparse_extrinsics[start_index:end_index]
                    pair_poses = naive_pair_poses
                    pair_point_cloud = sparse_all_pair_point_clous[start_index//4]
                    intrinsics, poses, pts_4_3dgs, color_4_3dgs, conf = sparse_intrinsics[start_index:end_index], pair_poses,  pair_point_cloud[:,:3], pair_point_cloud[:,3:], np.zeros_like(pair_point_cloud[:,:3])
                    pose_inv = np.linalg.inv(poses)
                    intrinsics[:,:1,:] = intrinsics[:,:1,:] * w_scaler
                    intrinsics[:,1:2,:] = intrinsics[:,1:2,:] * h_scaler

                    scale_gaussian = compute_scale_gaussian_by_project_pair_pcd(pts_4_3dgs, pose_inv, intrinsics)
                    local_gaussians.create_from_pcd_separate(pts_4_3dgs, color_4_3dgs, conf, scale_gaussian, 1.0)
                    if count%2==0 and pose_inv.shape[0]%2==0:
                        if pose_inv.shape[0]<4:
                            local_gaussians.init_RT_seq_pose(pose_inv, 2**level, pose_inv.shape[0])
                        elif is_power_of_two(count) or count%4==0:
                            local_gaussians.init_RT_seq_pose(pose_inv, 2**level, 2**(level+int(math.log2(count))))
                        elif pose_inv.shape[0]%4==0:
                            local_gaussians.init_RT_seq_pose(pose_inv, 2**level, 2**(level+1))                            
                    else:
                        local_gaussians.init_RT_seq_pose(pose_inv, 2**level, pose_inv.shape[0])
                    local_gaussians.training_setup(opt)
                    
                    for ii, pose in enumerate(pose_inv):
                        view_cam = scene.addTrainViewPose_progressive(None, pose)
                    
                    # w2c to cam
                    local_gaussians, scene = gaussian_training(local_gaussians, scene, start_index, end_index, pipe, bg, base_num_iters_gaussian_opt, opt, sh_up=True, out_img_folder=scene.model_path)
                    global_iter += base_num_iters_gaussian_opt
                    prev_level_list.append(local_gaussians)
        # level > 1
        else:
            print("length of prev_level_list", len(prev_level_list))
            for count in range(max(math.ceil(args.n_views/(2**level)), 1)):
                if count==0:
                    local_gaussian = prev_level_list[count]
                    masks=[]
                    all_views = scene.getTrainCameras().copy()
                    for ii, pose_tensor in enumerate(local_gaussian.P):
                        scene_index = ii+ count + 2**(level-1)
                        new_pose_tensor = pose_tensor
                        update_view_cam = all_views[scene_index]
                        if ii<4:
                            optimal_pose = optimize_pose(gaussians, update_view_cam, pipe, bg, new_pose_tensor.detach(), out_img_folder=scene.model_path)
                            gaussians.update_RT_seq_by_pose(optimal_pose, scene_index)
                            rendering = render(update_view_cam, gaussians, pipe, bg, camera_pose=optimal_pose)
                            depth = rendering["depth"]
                            alpha = rendering["alpha"]
                            mask = ((depth>0) & (alpha>alpha_threshold)).detach().cpu().numpy()
                            mask = mask.transpose((1, 2, 0))[:,:,:1].astype(np.uint8)
                            mask = cv2.resize(mask, (loww, lowh))
                            masks.append(mask)
                        else:
                            gaussians.update_RT_seq_by_pose(new_pose_tensor, scene_index)
                        print(gaussians.P.detach().cpu().numpy().shape)
                        print(np.array(masks).shape)
                        
                     
                    local_gaussian.reset_opacity()
                    xyz_trans = local_gaussian._xyz.clone()
                    rot_trans = local_gaussian._rotation.clone()
                    gaussians.add_local_gaussian(xyz_trans, rot_trans, local_gaussian, masks)
                    gaussians, scene = gaussian_training(gaussians, scene, count*(2**(level)), count*(2**(level)) + 2**(level), pipe, bg, base_num_iters_gaussian_opt*level, opt, sh_up=True, out_img_folder=scene.model_path)
                    global_iter += base_num_iters_gaussian_opt*level
                else:
                    cur_gaussian = prev_level_list[count*2-1]
                    if count*2 <= len(prev_level_list)-1:
                        print(f"merge gaussian {count*2-1} and {count*2}")
                        local_gaussian = prev_level_list[count*2]
                        masks=[]
                        all_views = scene.getTrainCameras().copy()
                        for ii, pose_tensor in enumerate(local_gaussian.P):
                            scene_index = ii+2**(level-1)
                            global_scene_index = scene_index + count*2**level
                            new_pose_tensor = pose_tensor
                            if ii<4:
                                optimal_pose = optimize_pose(cur_gaussian, all_views[global_scene_index], pipe, bg, new_pose_tensor.detach(), out_img_folder=scene.model_path)
                                cur_gaussian.update_RT_seq_by_pose(optimal_pose, scene_index)
                                rendering = render(all_views[scene_index], cur_gaussian, pipe, bg, camera_pose=optimal_pose)
                                depth = rendering["depth"]
                                alpha = rendering["alpha"]
                                mask = ((depth>0) & (alpha>alpha_threshold)).detach().cpu().numpy()
                                mask = mask.transpose((1, 2, 0))[:,:,:1].astype(np.uint8)
                                mask = cv2.resize(mask, (loww, lowh))
                                masks.append(mask)
                            else:
                                cur_gaussian.update_RT_seq_by_pose(new_pose_tensor, scene_index)
                        
                        local_gaussian.reset_opacity()
                        xyz_trans = local_gaussian._xyz.clone()
                        rot_trans = local_gaussian._rotation.clone()
                        cur_gaussian.add_local_gaussian(xyz_trans, rot_trans, local_gaussian, masks)
                        cur_gaussian, scene = gaussian_training(cur_gaussian, scene, count*(2**(level)), count*(2**(level)) + 2**(level), pipe, bg, base_num_iters_gaussian_opt*level, opt, sh_up=True, out_img_folder=scene.model_path)
                        global_iter += base_num_iters_gaussian_opt*level
                        cur_level_list.append(cur_gaussian)
                    else:
                        print(f"only gaussian {count*2-1}")
                        cur_level_list.append(cur_gaussian)
            del prev_level_list
            prev_level_list = cur_level_list
            del cur_level_list
            cur_level_list = []
            torch.cuda.empty_cache()
                
                
    if global_iter<opt.iterations:
        print("train globally")
        gaussians, scene = gaussian_training(gaussians, scene, count*(2**(level)), count*(2**(level)) + 2**(level), pipe, bg, opt.iterations-global_iter, opt, flag_global=True, sh_up=True, out_img_folder=scene.model_path)
        global_iter += opt.iterations-global_iter
    scene.save(global_iter)
    os.makedirs(scene.model_path + "pose/", exist_ok=True)
    train_cams = scene.getTrainCameras().copy()
    save_pose(scene.model_path + f"pose/pose_{global_iter}.npy", gaussians.P, train_cams)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--global_focal', type=float)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[500, 1000, 2000, 3000, 4000, 5000, 6000, 7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1000,3000, 5000, 7000, 10000, 20000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--n_views", type=int, default=None)
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--sample_random", action='store_true', default=False)
    # the below is temp  args
    parser.add_argument("--sparse_image_folder", type=str, default = None)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    os.makedirs(args.model_path, exist_ok=True)
    
    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    start_time = time.time()
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)
    print(time.time()-start_time)
    print("\nTraining complete.")
