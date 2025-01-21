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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import pickle
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


def render_set_optimize(model_path, name, iteration, views, gaussians, pipeline, background, args, source_image_path=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    raw_img_path = os.path.join(model_path, name, "ours_{}".format(iteration), "before_opt")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    os.makedirs(render_path, exist_ok=True)
    os.makedirs(raw_img_path, exist_ok=True)
    os.makedirs(gts_path, exist_ok=True)

    # Freeze Gaussian parameters
    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._opacity.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)

    train_poses = np.load(os.path.join(model_path, "pose/pose_{}.npy".format(iteration)))
    if source_image_path is not None:
        all_img_list = sorted(os.listdir(source_image_path))
        sample_rate = 2 if "Family" in source_image_path else 8
        ids = np.arange(len(all_img_list))
        i_test = ids[int(sample_rate / 2)::sample_rate]
        i_train = np.array([i for i in ids if i not in i_test])
        indices = np.linspace(0, len(i_train) - 1, args.n_views, dtype=int)
        new_i_train = [i_train[i] for i in indices]

    for ii in range(train_poses.shape[0]):
        train_poses[ii] = np.linalg.inv(train_poses[ii])  # c2w


    sparse_recon_res_path = os.path.join(source_image_path.replace("images",""), f"sparse/0/sparse_{args.n_views}.pkl")
    with open(sparse_recon_res_path, 'rb') as file:
        sparse_recon_res = pickle.load(file)
    sparse_extrinsics = sparse_recon_res["extrinsics"]
    sparse_intrinsics = sparse_recon_res["intrinsics"]
    visualizer(sparse_extrinsics, ["green" for _ in sparse_extrinsics], model_path + "pose/sparse_poses.png")

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        num_iter = args.optim_test_pose_iter
        patience = 100
        min_delta = 1e-5

        # Prepare initial camera pose
        smaller, larger = find_closest_numbers(new_i_train, i_test[view.uid])
        smaller_idx = new_i_train.index(smaller)
        larger_idx = new_i_train.index(larger)
        poses = generate_interpolated_path(poses=torch.tensor(np.stack(train_poses[smaller_idx:larger_idx + 1][:, :3, :])), n_interp=larger - smaller - 1)
        pose = poses[i_test[view.uid] - smaller - 1]
        
        c2w = torch.eye(4, dtype=torch.float32, device='cuda')
        c2w[:3, :3] = torch.tensor(pose[:3, :3], dtype=torch.float32, device='cuda')
        c2w[:3, 3] = torch.tensor(pose[:3, 3], dtype=torch.float32, device='cuda')
        w2c = c2w.inverse()
        camera_pose = get_tensor_from_camera(w2c)

        # Set up pose optimization
        camera_tensor_T = camera_pose[-3:].requires_grad_()
        camera_tensor_q = camera_pose[:4].requires_grad_()
        pose_optimizer = torch.optim.Adam(
            [
                {"params": [camera_tensor_T], "lr": 0.0003},
                {"params": [camera_tensor_q], "lr": 0.0003},
            ]
        )
        pose_optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(range(num_iter), desc=f"Tracking Time Step: {idx}", disable=True)

        # Keep track of best pose candidate and early stopping
        candidate_q = camera_tensor_q.clone().detach()
        candidate_T = camera_tensor_T.clone().detach()
        current_min_loss = float('inf')
        gt = view.original_image[0:3, :, :]
        no_improve_counter = 0  # Counter for early stopping

        for iteration in range(num_iter):
            # Rendering
            rendering = render(view, gaussians, pipeline, background, camera_pose=torch.cat([camera_tensor_q, camera_tensor_T]))["render"]
            
            if iteration == 0:
                torchvision.utils.save_image(rendering, os.path.join(raw_img_path, "{0:05d}".format(idx) + "_before_opt.png"))
            
            # # Calculate loss
            # loss = torch.abs(gt - rendering).mean()

            # # Early stopping check
            # if loss < current_min_loss - min_delta:
            #     current_min_loss = loss.item()
            #     candidate_q = camera_tensor_q.clone().detach()
            #     candidate_T = camera_tensor_T.clone().detach()
            #     no_improve_counter = 0  # Reset the counter if there's an improvement
            # else:
            #     no_improve_counter += 1

            # # Early stopping: if no improvement for 'patience' iterations, stop optimization early
            # if no_improve_counter >= patience:
            #     print(f"Early stopping at iteration {iteration} for view {idx} due to no significant improvement.")
            #     break

            # # Backpropagation and optimizer step
            # loss.backward()
            # pose_optimizer.step()
            # pose_optimizer.zero_grad(set_to_none=True)

            # if iteration % 10 == 0:
            #     print(f"Iteration {iteration}, Loss: {loss.item()}")

            progress_bar.update(1)

        # Use the best pose found
        camera_tensor_q = candidate_q
        camera_tensor_T = candidate_T
        progress_bar.close()

        # Render with optimized pose
        opt_pose = torch.cat([camera_tensor_q, camera_tensor_T])
        rendering_opt = render(view, gaussians, pipeline, background, camera_pose=opt_pose)["render"]

        # Save the optimized rendering
        torchvision.utils.save_image(
            rendering_opt, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
        )
        torchvision.utils.save_image(
            gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
        )


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

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")


    if not skip_test:
        render_set_optimize(
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
