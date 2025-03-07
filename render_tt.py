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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.pose_utils import get_tensor_from_camera
from utils.camera_utils import generate_interpolated_path
import numpy as np
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "dust3r")))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.dust3r_utils import  (compute_global_alignment, load_images, storePly, save_colmap_cameras, save_colmap_images, 
                                 round_python3, rigid_points_registration)
from utils.align_traj import align_scale_c2b_use_a2b, align_ate_c2b_use_a2b
from utils.camera_utils import visualizer

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        camera_pose = get_tensor_from_camera(view.world_view_transform.transpose(0, 1))
        rendering = render(
            view, gaussians, pipeline, background, camera_pose=camera_pose
        )["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(
            rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )

def find_closest_numbers(numbers, target):
    # Sort the list
    sorted_numbers = sorted(numbers)

    # Initialize variables to store the closest numbers
    smaller = None
    larger = None

    # Traverse through the sorted list to find the closest larger number
    for number in sorted_numbers:
        if number > target:
            larger = number
            break

    # Find the closest smaller number
    if sorted_numbers.index(larger) > 0:
        smaller = sorted_numbers[sorted_numbers.index(larger) - 1]

    return smaller, larger

def render_set_optimize(test_poses, model_path, name, iteration, views, gaussians, pipeline, background, args, source_image_path=None):
    render_path = os.path.join(model_path, name, f"ours_{iteration}", "renders")
    raw_img_path = os.path.join(model_path, name, f"ours_{iteration}", "before_opt")
    gts_path = os.path.join(model_path, name, f"ours_{iteration}", "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(raw_img_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    # Freeze the Gaussian parameters
    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    train_poses = np.load(os.path.join(model_path, f"pose/pose_{iteration}.npy"))

    # Invert the train poses
    for ii in range(train_poses.shape[0]):
        train_poses[ii] = np.linalg.inv(train_poses[ii])  # c2w

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        num_iter = args.optim_test_pose_iter
        pose = test_poses[idx]
        camera_pose = get_tensor_from_camera(pose)

        # Initialize pose parameters with gradients
        camera_tensor_T = camera_pose[-3:].detach().clone().requires_grad_()
        camera_tensor_q = camera_pose[:4].detach().clone().requires_grad_()

        # Setup optimizer
        pose_optimizer = torch.optim.Adam(
            [
                {"params": [camera_tensor_T], "lr": 0.0003},
                {"params": [camera_tensor_q], "lr": 0.0003},
            ]
        )
        pose_optimizer.zero_grad(set_to_none=True)

        # Early stopping parameters
        patience = 10     # Number of iterations to wait for improvement
        min_delta = 1e-4  # Minimum change in loss to qualify as an improvement
        no_improve_counter = 0

        # Keep track of best pose candidate
        candidate_q = camera_tensor_q.clone().detach()
        candidate_T = camera_tensor_T.clone().detach()
        current_min_loss = float('inf')

        gt = view.original_image[0:3, :, :].cuda()

        for iteration in range(num_iter):
            # Render the image with current pose
            rendering = render(
                view, gaussians, pipeline, background,
                camera_pose=torch.cat([camera_tensor_q, camera_tensor_T])
            )["render"]

            # Save the initial rendering and ground truth
            if iteration == 0:
                torchvision.utils.save_image(
                    rendering, os.path.join(raw_img_path, f"{idx:05d}_before_opt.png")
                )
                torchvision.utils.save_image(
                    gt, os.path.join(gts_path, f"{idx:05d}.png")
                )

            # Compute loss
            loss = torch.abs(gt - rendering).mean()

            # Check for early stopping
            if loss.item() < current_min_loss - min_delta:
                current_min_loss = loss.item()
                candidate_q = camera_tensor_q.clone().detach()
                candidate_T = camera_tensor_T.clone().detach()
                no_improve_counter = 0  # Reset counter
            else:
                no_improve_counter += 1

            if no_improve_counter >= patience:
                print(f"Early stopping at iteration {iteration} for view {idx} due to no significant improvement.")
                break

            if iteration % 10 == 0:
                print(f"View {idx}, Iteration {iteration}, Loss: {loss.item():.6f}")

            # Backpropagation and optimizer step
            loss.backward()
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)

            # Normalize quaternion to ensure it remains a valid rotation
            with torch.no_grad():
                camera_tensor_q /= torch.norm(camera_tensor_q)

        # Use the best pose found
        camera_tensor_q = candidate_q
        camera_tensor_T = candidate_T
        opt_pose = torch.cat([camera_tensor_q, camera_tensor_T])

        # Render with optimized pose
        rendering_opt = render(
            view, gaussians, pipeline, background, camera_pose=opt_pose
        )["render"]

        # Save the optimized rendering
        torchvision.utils.save_image(
            rendering_opt, os.path.join(render_path, f"{idx:05d}.png")
        )

    print("Rendering completed with early stopping optimization.")

def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    args,
):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, opt=args, shuffle=False)
        scene.progressive_index = args.n_views

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    img_base_path = dataset.source_path

    sparse_recon_res_path = os.path.join(args.model_path, f"sparse/0/sparse_{args.n_views}.pkl")
    with open(sparse_recon_res_path, 'rb') as file:
        sparse_recon_res = pickle.load(file)
    sparse_extrinsics = sparse_recon_res["extrinsics"]
    visualizer(sparse_extrinsics, None, dataset.model_path + "pose/sparse_poses.png")
    colmap_train_poses = []
    for view in scene.getTrainCameras():
        pose = view.world_view_transform.transpose(0, 1)
        colmap_train_poses.append(pose)

    colmap_test_poses = []
    for view in scene.getTestCameras():
        pose = view.world_view_transform.transpose(0, 1)
        colmap_test_poses.append(pose)

    
    colmap_train_poses = torch.stack(colmap_train_poses)
    colmap_test_poses = torch.stack(colmap_test_poses)
    origin = colmap_train_poses[0]
    colmap_train_poses = colmap_train_poses @ origin.inverse()
    colmap_test_poses = colmap_test_poses @ origin.inverse()

    visualizer(torch.cat([colmap_train_poses.cpu(), colmap_test_poses.cpu()]).numpy(), ["green" for _ in colmap_train_poses]+["red" for _ in colmap_test_poses], dataset.model_path + "pose/colmap_poses_train_test.png")
    train_poses = np.load(os.path.join(dataset.model_path,"pose/pose_{}.npy".format(iteration)))
    train_poses = torch.tensor(train_poses)
    test_poses_learned, scale_a2b = align_scale_c2b_use_a2b(colmap_train_poses, train_poses, colmap_test_poses)
    visualizer(torch.cat([train_poses,test_poses_learned.cpu()]).numpy(), ["green" for _ in train_poses]+["red" for _ in test_poses_learned], dataset.model_path + "pose/aligned_train_test.png")
    sparse_extrinsics = torch.tensor(sparse_extrinsics)


    if not skip_test:
        render_set_optimize(
            test_poses_learned,
            dataset.model_path,
            "test",
            scene.loaded_iter,
            scene.getTestCameras(),
            gaussians,
            pipeline,
            background,
            args,
            source_image_path = dataset.source_path+"/images"
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    
    parser.add_argument("--get_video", action="store_true")
    parser.add_argument("--n_views", default=None, type=int)
    parser.add_argument("--scene", default=None, type=str)
    parser.add_argument("--optim_test_pose_iter", default=500, type=int)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        args,
    )
