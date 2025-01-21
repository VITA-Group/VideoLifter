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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import scipy
import matplotlib.pyplot as plt
from matplotlib import cm
WARNED = False
import torch
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import scipy.interpolate as si


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]

    if cam_info.mask is not None:
        loaded_mask = PILtoTorch(cam_info.mask, resolution)
        loaded_mask = 1-loaded_mask
    else:
        loaded_mask = None

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def transform_poses_pca(poses):
    """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is negative
    if poses_recentered.mean(axis=0)[2, 1] < 0:
        poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
    poses_recentered[:, :3, 3] *= scale_factor
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

    return poses_recentered, transform

def generate_interpolated_path(poses, n_interp, spline_degree=5,
                               smoothness=.03, rot_weight=.1):
    """Creates a smooth spline path between input keyframe camera poses.

  Spline is calculated with poses in format (position, lookat-point, up-point).

  Args:
    poses: (n, 3, 4) array of input pose keyframes.
    n_interp: returned path will have n_interp * (n - 1) total poses.
    spline_degree: polynomial degree of B-spline.
    smoothness: parameter for spline smoothing, 0 forces exact interpolation.
    rot_weight: relative weighting of rotation/translation in spline solve.

  Returns:
    Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
  """

    def poses_to_points(poses, dist):
        """Converts from pose matrices to (position, lookat, up) format."""
        pos = poses[:, :3, -1]
        lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
        up = poses[:, :3, -1] + dist * poses[:, :3, 1]
        return np.stack([pos, lookat, up], 1)

    def points_to_poses(points):
        """Converts from (position, lookat, up) format to pose matrices."""
        return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])

    def interp(points, n, k, s):
        """Runs multidimensional B-spline interpolation on the input points."""
        sh = points.shape
        pts = np.reshape(points, (sh[0], -1))
        k = min(k, sh[0] - 1)
        tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
        u = np.linspace(0, 1, n, endpoint=False)
        new_points = np.array(scipy.interpolate.splev(u, tck))
        new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
        return new_points
    

    points = poses_to_points(poses, dist=rot_weight)
    new_points = interp(points,
                        n_interp * (points.shape[0] - 1),
                        k=spline_degree,
                        s=smoothness)
    return points_to_poses(new_points) 


def generate_nearby_poses(poses, idx=0, osci_scale=0.01, n_frames=120):
    ''' render_pose has to be of shape 1x3x4 '''
    nearby_views = []
    pose = poses[idx, ...][:3, :]
    for i in range(n_frames):
        oscillation_amount = np.sin(i / n_frames * 2*np.pi) * osci_scale

        translation_matrix = np.array([[0],
                                        [oscillation_amount],
                                        [0]])
        new_pose = pose.copy()
        new_pose[:, 3] += translation_matrix.flatten()
        nearby_views.append(new_pose)
    return np.stack(nearby_views)

def focus_point_fn(poses):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def generate_ellipse_path(poses, n_frames=120, const_speed=True, z_variation=0., z_phase=0., \
                          render_zoom_in=False, sc=20, aug_path=False, render_path=True, \
                          spiral_scale=0, render_start_view=34, render_dist_factor=1, osci_scale=0):
    """Generate an elliptical render path based on the given poses."""
    # Calculate the focal point for the path (cameras point toward this).
    center = focus_point_fn(poses[0:1]) # Nx3x4 -> 3

    if render_zoom_in:
        center[2] *= 0.6  # move y a little bit higher
        # Path height sits at z=0 (in middle of zero-mean capture pattern).
        offset = np.array([center[0], center[1], 0])  # 3,

        # Calculate the direction and distance between the first pose and center.
        idx = min(render_start_view, poses.shape[0]-1)
        direction = center - poses[idx, :3, 3]
        distance = np.linalg.norm(direction)
        direction /= distance

        # Generate the new positions by interpolating between the first pose and center.
        positions = np.linspace(0, render_dist_factor*distance, n_frames)[:, np.newaxis] * direction + poses[idx, :3, 3]

        # Perturb the positions with random noise.
        if spiral_scale > 0:
            spiral_theta = np.linspace(0, 2 * np.pi, n_frames)
            spiral_x = spiral_scale * np.cos(spiral_theta)
            spiral_y = spiral_scale * np.sin(spiral_theta)
            spiral_z = np.linspace(0, spiral_scale, n_frames)
            spiral_offset = np.stack([spiral_x, spiral_y, spiral_z], axis=1)
            positions += spiral_offset

        if osci_scale > 0:
            osci_amount = np.array([np.sin(x / n_frames * 5*np.pi) * osci_scale for x in range(n_frames)])
            offset = np.stack([np.zeros_like(osci_amount), osci_amount, np.zeros_like(osci_amount)], 1)
            positions += offset

        # Set path's up vector to axis closest to average of input pose up vectors.
        avg_up = poses[:, :3, 1].mean(0)
        avg_up = avg_up / np.linalg.norm(avg_up)  # (3,)
        ind_up = np.argmax(np.abs(avg_up))
        up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])  # (3,)

        # Generate the camera extrinsics using the provided viewmatrix function.
        extrinsics = np.stack([viewmatrix(p - center, up, p) for p in positions], axis=0)

        return extrinsics
    # Path height sits at z=0 (in middle of zero-mean capture pattern).
    offset = np.array([center[0], center[1], 0])

    # Calculate scaling for ellipse axes based on input camera positions.
    # the smaller, the closer to central object
    percentile = sc if (aug_path or render_path) else 90
    # percentile = sc if aug_path else 90
    sc = np.percentile(np.abs(poses[:, :3, 3] - offset), percentile, axis=0) # (3,)
    # Use ellipse that is symmetric about the focal point in xy.
    low = -sc + offset
    high = sc + offset
    # Optional height variation need not be symmetric
    z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)
    z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)

    def get_positions(theta):
        # Interpolate between bounds with trig functions to get ellipse in x-y.
        # Optionally also interpolate in z to change camera height along path.
        return np.stack([
            low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),
            low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),
            z_variation * (z_low[2] + (z_high - z_low)[2] *
                           (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
        ], -1)

    theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)
    positions = get_positions(theta) # (121, 3)

    if const_speed:
        # Resample theta angles so that the velocity is closer to constant.
        lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
        theta = stepfun.sample_np(None, theta, np.log(lengths), n_frames + 1)
        positions = get_positions(theta)

    # Throw away duplicated last position.
    positions = positions[:-1]

    # Set path's up vector to axis closest to average of input pose up vectors.
    avg_up = poses[:, :3, 1].mean(0)
    avg_up = avg_up / np.linalg.norm(avg_up)
    ind_up = np.argmax(np.abs(avg_up))
    up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])
    return np.stack([viewmatrix(p - center, up, p) for p in positions])


def viewmatrix(lookdir, up, position):
    """Construct lookat view matrix."""
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m

def normalize(x):
    """Normalization helper function."""
    return x / np.linalg.norm(x)

def pad_poses(p):
    """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
    bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)
    return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
    """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
    return p[..., :3, :4]


def visualizer(camera_poses, colors=None, save_path="/mnt/data/1.png"):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if colors==None:
        num_poses = len(camera_poses)
        colormap = cm.get_cmap("viridis", num_poses)
        colors = colormap(np.linspace(0, 1, num_poses))
    for pose, color in zip(camera_poses, colors):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]  # Corrected to use 3D translation component
        camera_positions = np.einsum(
            "...ij,...j->...i", np.linalg.inv(rotation), -translation
        )

        ax.scatter(
            camera_positions[0],
            camera_positions[1],
            camera_positions[2],
            c=color,
            marker="o",
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera Poses")

    plt.savefig(save_path)
    plt.close()

    return save_path


def slerp(quat0, quat1, t):
    """Spherical linear interpolation between two quaternions."""
    dot = np.dot(quat0, quat1)
    if dot < 0.0:
        quat1 = -quat1
        dot = -dot
    if dot > 0.95:
        return quat0 + t * (quat1 - quat0)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    sin_theta_t = np.sin(theta_0 * t)
    sin_theta_1 = np.sin((1.0 - t) * theta_0)
    s0 = sin_theta_1 / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    return (s0 * quat0) + (s1 * quat1)

def rotation_matrix_to_quaternion(matrix):
    """Convert a rotation matrix to a quaternion."""
    m = matrix[:3, :3]
    t = np.trace(m)
    if t > 0.0:
        t = np.sqrt(t + 1.0)
        w = 0.5 * t
        t = 0.5 / t
        x = (m[2, 1] - m[1, 2]) * t
        y = (m[0, 2] - m[2, 0]) * t
        z = (m[1, 0] - m[0, 1]) * t
    else:
        i = np.argmax(np.diagonal(m))
        j = (i + 1) % 3
        k = (i + 2) % 3
        t = np.sqrt(m[i, i] - m[j, j] - m[k, k] + 1.0)
        q = np.zeros(4)
        q[i] = 0.5 * t
        t = 0.5 / t
        q[3] = (m[k, j] - m[j, k]) * t
        q[j] = (m[j, i] + m[i, j]) * t
        q[k] = (m[k, i] + m[i, k]) * t
        x, y, z, w = q
    return np.array([x, y, z, w])



def generate_zoomin_trajectory(extrinsics, num_steps, num_rotations=2, spiral_rad_w=0.2, fwd_dist_w=3):
    pose_A = extrinsics[0]  # Camera A
    pose_B = extrinsics[1]  # Camera B

    pos_A = pose_A[:3, 3]
    pos_B = pose_B[:3, 3]
    forward_A = -pose_A[:3, 2]  # Assuming forward is the negative z-axis

    center_point = (pos_A + pos_B) / 2
    radius = np.linalg.norm(pos_A - pos_B) * spiral_rad_w

    # Calculate the forward position significantly ahead in the direction of A's negative z-axis
    forward_dist = np.linalg.norm(pos_A - pos_B) * fwd_dist_w  # Adjust the scaling as needed
    forward_point = pos_A + forward_dist * forward_A

    # Compute total number of steps for each segment
    num_forward_steps = num_steps // 2
    num_backward_steps = num_steps - num_forward_steps

    # Generate the spiral trajectory
    positions = []
    theta_step = 2 * np.pi * num_rotations / num_steps
    for i in range(num_steps):
        theta = i * theta_step
        if i < num_forward_steps:
            # Spiral from A to forward point
            fraction = i / num_forward_steps
            fraction_radius = i / num_forward_steps
            target_point = forward_point
            start_point = pos_A
        else:
            # Spiral from forward point to B
            fraction = (i - num_forward_steps) / num_backward_steps
            fraction_radius = (num_steps - i) / num_backward_steps
            target_point = pos_B
            start_point = forward_point

        x_offset = radius * fraction_radius * np.cos(theta)
        y_offset = radius * fraction_radius * np.sin(theta)
        position = np.array([
            start_point[0] + fraction * (target_point[0] - start_point[0]) + x_offset,
            start_point[1] + fraction * (target_point[1] - start_point[1]) + y_offset,
            start_point[2] + fraction * (target_point[2] - start_point[2])
        ])
        positions.append(position)

    positions = np.array(positions)

    # Quaternion interpolation for rotations
    quat_A = rotation_matrix_to_quaternion(pose_A[:3, :3])
    quat_B = rotation_matrix_to_quaternion(pose_B[:3, :3])
    quaternions = np.array([slerp(quat_A, quat_B, t) for t in np.linspace(0, 1, num_steps)])

    # Combine positions and rotations into a trajectory
    trajectory = np.zeros((num_steps, 4, 4))
    for i in range(num_steps):
        trajectory[i, :3, :3] = quaternion_to_rotation_matrix(quaternions[i])
        trajectory[i, :3, 3] = positions[i]
        trajectory[i, 3, 3] = 1

    return trajectory

def quaternion_to_rotation_matrix(quat):
    """Convert a quaternion to a rotation matrix."""
    x, y, z, w = quat
    xx, xy, xz = x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z
    wx, wy, wz = w * x, w * y, w * z

    matrix = np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
    return matrix


def interp_poses_bspline(c2ws, N_novel_imgs, input_times, degree):
    target_trans = torch.tensor(scipy_bspline(
        c2ws[:, :3, 3], n=N_novel_imgs, degree=degree, periodic=False).astype(np.float32)).unsqueeze(2)
    rots = R.from_matrix(c2ws[:, :3, :3])
    slerp = Slerp(input_times, rots)
    target_times = np.linspace(input_times[0], input_times[-1], N_novel_imgs)
    target_rots = torch.tensor(
        slerp(target_times).as_matrix().astype(np.float32))
    target_poses = torch.cat([target_rots, target_trans], dim=2)
    target_poses = convert3x4_4x4(target_poses)
    return target_poses

def scipy_bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Closed curve
    if periodic:
        kv = np.arange(-degree, count+degree+1)
        factor, fraction = divmod(count+degree+1, count)
        cv = np.roll(np.concatenate(
            (cv,) * factor + (cv[:fraction],)), -1, axis=0)
        degree = np.clip(degree, 1, degree)

    # Opened curve
    else:
        degree = np.clip(degree, 1, count-1)
        kv = np.clip(np.arange(count+degree+1)-degree, 0, count-degree)

    # Return samples
    max_param = count - (degree * (1-periodic))
    spl = si.BSpline(kv, cv, degree)
    return spl(np.linspace(0, max_param, n))

def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(
                input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor(
                [[0, 0, 0, 1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
    else:
        if len(input.shape) == 3:
            output = np.concatenate(
                [input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate(
                [input, np.array([[0, 0, 0, 1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output