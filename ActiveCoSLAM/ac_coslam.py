import os

# os.environ['TCNN_CUDA_ARCHITECTURES'] = '86'
import time
# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import shutil
import json
import cv2

from torch.utils.data import DataLoader
# from tqdm import tqdm
from tqdm import tqdm
from tqdm.rich import trange

# Local imports
import config

from ActiveCoSLAM.ac_scene_rep import JointEncoding
from ActiveCoSLAM.information_gain import choose_new_k, get_rays_np
from ActiveCoSLAM.ac_dataset import get_dataset

# from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation
from optimization.utils import (
    at_to_transform_matrix,
    qt_to_transform_matrix,
    matrix_to_axis_angle,
    matrix_to_quaternion,
)


class CoSLAM:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)  # 初始化关键帧数据库 长度 119

        self.model = JointEncoding(config, self.bounding_box).to(self.device)
        self.downsample_rate = self.config["active"]["downsample_rate"]

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def get_pose_representation(self):
        """
        Get the pose representation axis-angle or quaternion
        """
        if self.config["training"]["rot_rep"] == "axis_angle":
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print(
                "Using axis-angle as rotation representation, identity init would cause inf"
            )

        elif self.config["training"]["rot_rep"] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError

    def create_pose_data(self):
        """
        Create the pose data
        """
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
        self.load_gt_pose()

    def create_bounds(self):
        """
        Get the pre-defined bounds for the scene
        """
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(
            self.device)

    def create_kf_database(self, config):
        """
        Create the keyframe database
        """
        num_kf = int(
            # self.config["mapping"]["keyframe_every"] = 5 每五帧取一帧
            # self.dataset.num_frames = 592
            # +1 应该是第一帧

            self.dataset.num_frames // self.config["mapping"]["keyframe_every"] + 1
        )
        print("#kf: ", num_kf)
        # num_rays_to_save= total pixel *  5% (下采样)
        print("#Pixels to save: ", self.dataset.num_rays_to_save)
        return KeyFrameDatabase(
            config,
            self.dataset.H,
            self.dataset.W,
            num_kf,
            self.dataset.num_rays_to_save,
            self.device,
        )

    def load_gt_pose(self):
        """
        Load the ground truth pose
        """
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[i] = pose

    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))

    def save_ckpt(self, save_path):
        """
        Save the model parameters and the estimated pose
        """
        save_dict = {
            "pose": self.est_c2w_data,
            "pose_rel": self.est_c2w_data_rel,
            "model": self.model.state_dict(),
        }
        torch.save(save_dict, save_path)
        print("Save the checkpoint")

    def load_ckpt(self, load_path):
        """
        Load the model parameters and the estimated pose
        """
        dict = torch.load(load_path)
        self.model.load_state_dict(dict["model"])
        self.est_c2w_data = dict["pose"]
        self.est_c2w_data_rel = dict["pose_rel"]

    def select_samples(self, H, W, samples):
        """
        randomly select samples from the image
        #假设图像 800*800, samples=2048
        # indice就是从 0 到 640000 之间随机选取 2048 个数作为 index
        """
        # indice = torch.randint(H*W, (samples,))
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice

    def get_loss_from_ret(
            self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False
    ):
        """
        Get the training loss
        """
        loss = 0
        if rgb:  # 权重 1.0
            loss += self.config["training"]["rgb_weight"] * ret["rgb_loss"]
        if depth:  # 权重 0.1
            loss += self.config["training"]["depth_weight"] * ret["depth_loss"]
        if sdf:  # 权重 5000
            loss += self.config["training"]["sdf_weight"] * ret["sdf_loss"]
        if fs:  # 权重 10
            loss += self.config["training"]["fs_weight"] * ret["fs_loss"]

        if smooth and self.config["training"]["smooth_weight"] > 0:  # 权重 1e-8
            loss += self.config["training"]["smooth_weight"] * self.smoothness(
                self.config["training"]["smooth_pts"],
                self.config["training"]["smooth_vox"],
                margin=self.config["training"]["smooth_margin"],
            )

        return loss

    def first_frame_mapping(self, batch, n_iters=100):
        """
        First frame mapping, 把第一帧的位姿保存给 est_c2w_data和est_c2w_data_rel
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float

        """
        # ********************* 读取第0帧的相机位姿 *********************

        print("First frame mapping...")
        c2w = batch["c2w"][0].to(self.device)
        self.est_c2w_data[0] = c2w  # [4,4]
        self.est_c2w_data_rel[0] = c2w  # 第0帧的观测位姿 直接作为 位姿估计

        self.model.train()
        # 如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。
        # 对于BN层: 
        # model.train()是保证BN层用每一批数据的均值和方差
        # model.eval()是保证BN用全部训练数据的均值和方差；
        # 对于Dropout:
        # model.train()是随机取一部分网络连接来训练更新参数，
        # model.eval()是利用到了所有网络连接。

        # Training
        for i in trange(n_iters):  # 1000
            # ********************* 获得第0帧每个像素的颜色，深度，方向 *********************
            # print(f"\r{i}/{n_iters}", end="")

            self.map_optimizer.zero_grad()
            indice = self.select_samples(
                # 从一个范围内的整数（0 到 H * W - 1）中随机选择samples=2048个样本像素
                # indice就是从 0 到 182528 之间随机选取 2048 个数作为 index
                self.dataset.H,
                self.dataset.W,
                self.config["mapping"]["sample"],  # 2048
            )
            # 每个像素在图像上的位置索引
            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            # 计算的是每个索引在高度（行）方向上的位置，即行索引indice_h。 
            # 得到每个采样得到的像素的h,w值[2048]
            rays_d_cam = (batch["direction"].squeeze(0)[indice_h, indice_w, :].to(self.device))
            # 得到每个样本像素的方向，作为目标射线方向    [2048,3]
            target_s = batch["rgb"].squeeze(0)[indice_h, indice_w, :].to(self.device)
            # 得到每个样本像素的颜色，作为目标颜色        [2048.3]
            target_d = (batch["depth"].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(
                -1))
            # 得到每个样本像素的深度，作为目标深度        [2048,1]

            rays_o = c2w[None, :3, -1].repeat(self.config["mapping"]["sample"], 1)
            # 世界坐标系下的射线原点，即变换矩阵的t，即相机位置 [2048,3]
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)
            # rays_d_cam[..., None, :] 相机坐标系中的射线方向： [2048,1,3]
            # c2w[:3, :3] : 旋转矩阵    [3,3]
            # sum(x,-1) 在最后一个维度上进行求和

            # Forward
            # ********************* 前向传播: 得到rgb图，深度图，rgb损失，深度损失,sdf损失，fs损失 *********************

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            # ********************* 反响传播: 优化encoder/decoder网络的参数 *********************

            loss.backward()
            self.map_optimizer.step()

        # First frame will always be a keyframe
        # ********************* 将当前帧加入关键帧 *********************
        self.keyframeDatabase.add_keyframe(
            batch, filter_depth=self.config["mapping"]["filter_depth"]
        )
        if self.config["mapping"]["first_mesh"]:
            self.save_mesh(0)

        print("First frame mapping done")
        return ret, loss

    def current_frame_mapping(self, batch, cur_frame_id):
        """
        Current frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float

        """
        if self.config["mapping"]["cur_frame_iters"] <= 0:
            # self.config['mapping']['cur_frame_iters']= 0
            # 所以current_frame_mapping不会执行
            return
        print("Current frame mapping...")
        # ********************* 读取当前帧的位姿估计 *********************

        c2w = self.est_c2w_data[cur_frame_id].to(self.device)

        self.model.train()

        # Training
        for i in range(self.config["mapping"]["cur_frame_iters"]):
            self.cur_map_optimizer.zero_grad()
            indice = self.select_samples(
                self.dataset.H, self.dataset.W, self.config["mapping"]["sample"]  # 2048
            )

            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = (
                batch["direction"].squeeze(0)[indice_h, indice_w, :].to(self.device)
            )
            target_s = batch["rgb"].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = (
                batch["depth"]
                .squeeze(0)[indice_h, indice_w]
                .to(self.device)
                .unsqueeze(-1)
            )

            rays_o = c2w[None, :3, -1].repeat(self.config["mapping"]["sample"], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.cur_map_optimizer.step()

        return ret, loss

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        """
        Smoothness loss of feature grid
        """
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points - 1) * voxel_size
        offset_max = (
                self.bounding_box[:, 1] - self.bounding_box[:, 0] - grid_size - 2 * margin
        )

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, "cpu", flatten=False).float().to(volume)
        pts = (
                (coords + torch.rand((1, 1, 1, 3)).to(volume)) * voxel_size
                + self.bounding_box[:, 0]
                + offset
        )

        if self.config["grid"]["tcnn_encoding"]:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (
                    self.bounding_box[:, 1] - self.bounding_box[:, 0]
            )

        # query_sdf 返回 sdf, h,beta
        # 原始 COSLAM sdf是 63*63*63*32
        sdf = self.model.query_sdf(pts_tcnn, embed=True)  # 63, 63, 63, 32

        # 在这段代码中，需要计算相邻点之间的差异。
        # sdf[1:, ...]表示第二个点（1...n） - sdf[:-1, ...]表示第一个点（0...n-1）
        # 如果从0开始，那么第一个点没有前一个点可以比较，会导致索引错误
        tv_x = torch.pow(sdf[1:, ...] - sdf[:-1, ...], 2).sum()
        tv_y = torch.pow(sdf[:, 1:, ...] - sdf[:, :-1, ...], 2).sum()
        tv_z = torch.pow(sdf[:, :, 1:, ...] - sdf[:, :, :-1, ...], 2).sum()
        # Edited最后一列是beta，不需要所以到-1
        # tv_x = torch.pow(sdf[1:, :-1] - sdf[:-1, :-1], 2).sum()
        # tv_y = torch.pow(sdf[:, 1:, :-1] - sdf[:, :-1, :-1], 2).sum()
        # tv_z = torch.pow(sdf[:, :, 1:, :-1] - sdf[:, :, :-1, :-1], 2).sum()

        loss = (tv_x + tv_y + tv_z) / (sample_points ** 3)

        return loss

    def get_pose_param_optim(self, poses, mapping=True):
        task = "mapping" if mapping else "tracking"
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))
        pose_optimizer = torch.optim.Adam(
            [
                {"params": cur_rot, "lr": self.config[task]["lr_rot"]},
                {"params": cur_trans, "lr": self.config[task]["lr_trans"]},
            ]
        )

        return cur_rot, cur_trans, pose_optimizer

    def global_BA(self, batch, cur_frame_id):
        """
        Global bundle adjustment that includes all the keyframes and the current frame
        包含了 RaySampling
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        """
        pose_optimizer = None
        # 获取所有关键帧的位姿
        # all the KF poses: 0, 5, 10, ... 每五个取一个
        poses = torch.stack(
            [
                self.est_c2w_data[i]
                for i in range(
                0, cur_frame_id, self.config["mapping"]["keyframe_every"]
            )
            ]
        )

        # 获取所有关键帧的位姿和 id
        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(
            list(range(0, cur_frame_id, self.config["mapping"]["keyframe_every"]))
        )

        if len(self.keyframeDatabase.frame_ids) < 2:
            # 如果关键帧库的数量小于 2, 直接使用这些位姿,不做挑选
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None, ...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)

        else:
            # 如果大于 2, 固定第一帧,对后面的进行优化
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None, ...]

            if self.config["mapping"]["optim_cur"]:  # 是否优化当前帧
                # 对于 TUM : True
                (
                    cur_rot,
                    cur_trans,
                    pose_optimizer,
                ) = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                # 如果不优化当前帧, 就只加入 poses_all中就行了
                (
                    cur_rot,
                    cur_trans,
                    pose_optimizer,
                ) = self.get_pose_param_optim(poses[1:])
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)

        # Set up optimizer
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        current_rays = torch.cat(
            [batch["direction"], batch["rgb"], batch["depth"][..., None]], dim=-1
        )
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        # self.config['mapping']['iters'] = 20
        # "mapping" "sample"= 2048
        for i in range(self.config["mapping"]["iters"]):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            # sample_global_rays 对应论文中的全局采样即对所有的关键帧采样
            # 而不是像 niceslam 一样维护一个关键帧 list
            rays, ids = self.keyframeDatabase.sample_global_rays(
                self.config["mapping"]["sample"]
            )

            # TODO: Checkpoint...
            idx_cur = random.sample(
                range(0, self.dataset.H * self.dataset.W),
                max(
                    self.config["mapping"]["sample"]
                    // len(self.keyframeDatabase.frame_ids),
                    self.config["mapping"]["min_pixels_cur"],
                ),
            )
            current_rays_batch = current_rays[idx_cur, :]

            rays = torch.cat([rays, current_rays_batch], dim=0)  # N, 7
            ids_all = torch.cat(
                [
                    ids // self.config["mapping"]["keyframe_every"],
                    -torch.ones((len(idx_cur))),
                ]
            ).to(torch.int64)

            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(
                rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1
            )
            rays_o = (
                poses_all[ids_all, None, :3, -1]
                .repeat(1, rays_d.shape[1], 1)
                .reshape(-1, 3)
            )
            rays_d = rays_d.reshape(-1, 3)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            loss = self.get_loss_from_ret(ret, smooth=True)

            loss.backward(retain_graph=True)

            if (i + 1) % cfg["mapping"]["map_accum_step"] == 0:

                if (i + 1) > cfg["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print("Wait update")
                self.map_optimizer.zero_grad()

            if (
                    pose_optimizer is not None
                    and (i + 1) % cfg["mapping"]["pose_accum_step"] == 0
            ):
                # 姿态优化器在每个 pose_accum_step(5步)之后更新一次姿态参数。
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                # 计算新的位姿矩阵
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # So current pose is always unchanged
                if self.config["mapping"]["optim_cur"]:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

                else:
                    current_pose = self.est_c2w_data[cur_frame_id][None, ...]
                    # SE3 poses

                    poses_all = torch.cat(
                        [poses_fixed, pose_optim, current_pose], dim=0
                    )

                # zero_grad here
                pose_optimizer.zero_grad()
        # 在循环结束后，如果存在姿态优化器且有多于一个帧的姿态数据时，进行更新
        if pose_optimizer is not None and len(frame_ids_all) > 1:
            # 更新所有关键帧的姿态，这是在整个优化过程结束后对关键帧姿态进行最终调整
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i + 1].item())] = (
                    self.matrix_from_tensor(cur_rot[i: i + 1], cur_trans[i: i + 1])
                    .detach()
                    .clone()[0]
                )

            if self.config["mapping"]["optim_cur"]:
                print("Update current pose")
                self.est_c2w_data[cur_frame_id] = (
                    self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:])
                    .detach()
                    .clone()[0]
                )
        # 循环内部的更新是一个逐步的优化过程，用于持续调整姿态估计。
        #   这些更新有助于在整个映射过程中不断改进姿态估计
        # 循环外的更新是在整个映射过程结束后进行的最终调整
        #   它确保了所有关键帧和当前帧的姿态估计都是最新和最准确的

    def predict_current_pose(self, frame_id, constant_speed=True):
        """
        Predict current pose from previous pose using camera motion model
        """
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id - 1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev

        else:
            c2w_est_prev_prev = self.est_c2w_data[frame_id - 2].to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id - 1].to(self.device)
            delta = c2w_est_prev @ c2w_est_prev_prev.float().inverse()
            self.est_c2w_data[frame_id] = delta @ c2w_est_prev

        return self.est_c2w_data[frame_id]

    def tracking_pc(self, batch, frame_id):
        """
        Tracking camera pose of current frame using point cloud loss
        (Not used in the paper, but might be useful for some cases)
        """

        c2w_gt = batch["c2w"][0].to(self.device)

        cur_c2w = self.predict_current_pose(
            frame_id, self.config["tracking"]["const_speed"]
        )

        cur_trans = torch.nn.parameter.Parameter(cur_c2w[..., :3, 3].unsqueeze(0))
        cur_rot = torch.nn.parameter.Parameter(
            self.matrix_to_tensor(cur_c2w[..., :3, :3]).unsqueeze(0)
        )
        pose_optimizer = torch.optim.Adam(
            [
                {"params": cur_rot, "lr": self.config["tracking"]["lr_rot"]},
                {"params": cur_trans, "lr": self.config["tracking"]["lr_trans"]},
            ]
        )
        best_sdf_loss = None

        iW = self.config["tracking"]["ignore_edge_W"]
        iH = self.config["tracking"]["ignore_edge_H"]

        thresh = 0
        # iter_point=0 不执行这一段
        if self.config["tracking"]["iter_point"] > 0:
            indice_pc = self.select_samples(
                self.dataset.H - iH * 2,
                self.dataset.W - iW * 2,
                self.config["tracking"]["pc_samples"],
            )
            rays_d_cam = (
                batch["direction"][:, iH:-iH, iW:-iW]
                .reshape(-1, 3)[indice_pc]
                .to(self.device)
            )
            target_s = (
                batch["rgb"][:, iH:-iH, iW:-iW]
                .reshape(-1, 3)[indice_pc]
                .to(self.device)
            )
            target_d = (
                batch["depth"][:, iH:-iH, iW:-iW]
                .reshape(-1, 1)[indice_pc]
                .to(self.device)
            )

            valid_depth_mask = ((target_d > 0.0) * (target_d < 5.0))[:, 0]

            rays_d_cam = rays_d_cam[valid_depth_mask]
            target_s = target_s[valid_depth_mask]
            target_d = target_d[valid_depth_mask]

            for i in range(self.config["tracking"]["iter_point"]):  # for i in range(0)
                pose_optimizer.zero_grad()
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                rays_o = c2w_est[..., :3, -1].repeat(len(rays_d_cam), 1)
                rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)
                pts = rays_o + target_d * rays_d

                pts_flat = (pts - self.bounding_box[:, 0]) / (
                        self.bounding_box[:, 1] - self.bounding_box[:, 0]
                )

                out = self.model.query_color_sdf_beta(pts_flat)

                sdf = out[:, -1]
                rgb = torch.sigmoid(out[:, :3])

                # TODO: Change this
                loss = 5 * torch.mean(torch.square(rgb - target_s)) + 1000 * torch.mean(
                    torch.square(sdf)
                )

                if best_sdf_loss is None:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()

                with torch.no_grad():
                    c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                    if loss.cpu().item() < best_sdf_loss:
                        best_sdf_loss = loss.cpu().item()
                        best_c2w_est = c2w_est.detach()
                        thresh = 0
                    else:
                        thresh += 1
                if thresh > self.config["tracking"]["wait_iters"]:
                    break

                loss.backward()
                pose_optimizer.step()

        if self.config["tracking"]["best"]:  # False
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

        if frame_id % self.config["mapping"]["keyframe_every"] != 0:
            # Not a keyframe, need relative pose
            kf_id = frame_id // self.config["mapping"]["keyframe_every"]
            kf_frame_id = kf_id * self.config["mapping"]["keyframe_every"]
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta
        print(
            "Best loss: {}, Camera loss{}".format(
                F.l1_loss(best_c2w_est.to(self.device)[0, :3], c2w_gt[:3]).cpu().item(),
                F.l1_loss(c2w_est[0, :3], c2w_gt[:3]).cpu().item(),
            )
        )

    def tracking_render(self, batch, frame_id):
        """
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        """

        c2w_gt = batch["c2w"][0].to(self.device)

        # Initialize current pose
        if self.config["tracking"]["iter_point"] > 0:  # iter_point=0
            cur_c2w = self.est_c2w_data[frame_id]
        else:
            cur_c2w = self.predict_current_pose(
                frame_id, self.config["tracking"]["const_speed"]  # Ture
            )

        indice = None
        best_sdf_loss = None
        thresh = 0

        iW = self.config["tracking"]["ignore_edge_W"]
        iH = self.config["tracking"]["ignore_edge_H"]

        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(
            cur_c2w[None, ...], mapping=False
        )

        # Start tracking
        for i in range(self.config["tracking"]["iter"]):
            pose_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

            # Note here we fix the sampled points for optimisation
            if indice is None:
                indice = self.select_samples(
                    self.dataset.H - iH * 2,
                    self.dataset.W - iW * 2,
                    self.config["tracking"]["sample"],
                )

                # Slicing
                indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (
                        self.dataset.H - iH * 2
                )
                rays_d_cam = (
                    batch["direction"]
                    .squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :]
                    .to(self.device)
                )
            target_s = (
                batch["rgb"]
                .squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :]
                .to(self.device)
            )
            target_d = (
                batch["depth"]
                .squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w]
                .to(self.device)
                .unsqueeze(-1)
            )

            rays_o = c2w_est[..., :3, -1].repeat(self.config["tracking"]["sample"], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)

            if best_sdf_loss is None:
                best_sdf_loss = loss.cpu().item()
                best_c2w_est = c2w_est.detach()

            with torch.no_grad():
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                if loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    thresh = 0
                else:
                    thresh += 1

            if thresh > self.config["tracking"]["wait_iters"]:
                break

            loss.backward()
            pose_optimizer.step()

        if self.config["tracking"]["best"]:
            # Use the pose with smallest loss
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            # Use the pose after the last iteration
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

        # Save relative pose of non-keyframes
        if frame_id % self.config["mapping"]["keyframe_every"] != 0:
            kf_id = frame_id // self.config["mapping"]["keyframe_every"]
            kf_frame_id = kf_id * self.config["mapping"]["keyframe_every"]
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta

        print(
            "Best loss: {:.4f}, Last loss: {:.4f}".format(
                F.l1_loss(best_c2w_est.to(self.device)[0, :3], c2w_gt[:3]).cpu().item(),
                F.l1_loss(c2w_est[0, :3], c2w_gt[:3]).cpu().item(),
            )
        )

    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.config["mapping"]["keyframe_every"] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config["mapping"]["keyframe_every"]
                kf_frame_id = kf_id * self.config["mapping"]["keyframe_every"]
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i]
                poses[i] = delta @ c2w_key

        return poses

    def create_optimizer(self):
        """
        Create optimizer for mapping
        """
        # Optimizer for BA
        trainable_parameters = [
            {
                "params": self.model.decoder.parameters(),
                "weight_decay": 1e-6,
                "lr": self.config["mapping"]["lr_decoder"],
            },
            {
                "params": self.model.embed_fn.parameters(),
                "eps": 1e-15,
                "lr": self.config["mapping"]["lr_embed"],
            },
        ]

        if not self.config["grid"]["oneGrid"]:
            trainable_parameters.append(
                {
                    "params": self.model.embed_fn_color.parameters(),
                    "eps": 1e-15,
                    "lr": self.config["mapping"]["lr_embed_color"],
                }
            )

        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))

        # Optimizer for current frame mapping
        if self.config["mapping"]["cur_frame_iters"] > 0:
            params_cur_mapping = [
                {
                    "params": self.model.embed_fn.parameters(),
                    "eps": 1e-15,
                    "lr": self.config["mapping"]["lr_embed"],
                }
            ]
            if not self.config["grid"]["oneGrid"]:
                params_cur_mapping.append(
                    {
                        "params": self.model.embed_fn_color.parameters(),
                        "eps": 1e-15,
                        "lr": self.config["mapping"]["lr_embed_color"],
                    }
                )

            self.cur_map_optimizer = optim.Adam(params_cur_mapping, betas=(0.9, 0.99))

    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(
            self.config["data"]["output"],
            self.config["data"]["exp_name"],
            "mesh_track{}.ply".format(i),
        )
        if self.config["mesh"]["render_color"]:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color
        extract_mesh(
            self.model.query_sdf,
            self.config,
            self.bounding_box,
            color_func=color_func,
            marching_cube_bound=self.marching_cube_bound,
            voxel_size=voxel_size,
            mesh_savepath=mesh_savepath,
        )

    def run(self):
        self.create_optimizer()
        dataset_size = len(self.dataset)
        # 我想将 self.dataset 中的数据的前百分之 25的图片保存为训练集
        # 后百分之 75的图片保存为验证集

        train_size = int(0.3 * dataset_size)  # 前百分之 30的图片作为训练集
        holdout_size = dataset_size - train_size  # 后百分之 70的图片作为保留集
        # 按顺序划分
        train_dataset = torch.utils.data.Subset(self.dataset, range(0, train_size))
        holdout_dataset = torch.utils.data.Subset(self.dataset, range(train_size, dataset_size))

        all_data_loader = DataLoader(self.dataset, num_workers=self.config["data"]["num_workers"])
        train_loader = DataLoader(train_dataset, num_workers=self.config["data"]["num_workers"])
        validation_loader = DataLoader(holdout_dataset, num_workers=self.config["data"]["num_workers"])
        i_train = train_dataset.indices
        i_holdout = holdout_dataset.indices

        all_images = []
        for batch in all_data_loader:
            all_images.append(batch['rgb'].detach().cpu().numpy())  # 确保只在CPU上操作，避免GPU溢出
            # Start Co-SLAM!
        all_images = np.concatenate(all_images, axis=0)  # 最后再进行一次合并
        all_images = torch.Tensor(all_images).to(self.device)

        for i, batch in tqdm(enumerate(train_loader)):
            # batch = {
            # 'frame_id': [1],
            # 'c2w': [1, 4, 4],
            # 'rgb': [1, H, W, 3], 1*368*496*3
            # 'depth': [1, H, W, 1], 1*368*496
            # 'direction': [1, H, W, 3] 1*368*496*3
            # }

            # ******** Visualisation *************
            if self.config["mesh"]["visualisation"]:
                rgb = cv2.cvtColor(
                    batch["rgb"].squeeze().cpu().numpy(), cv2.COLOR_BGR2RGB
                )
                raw_depth = batch["depth"]
                mask = (raw_depth >= self.config["cam"]["depth_trunc"]).squeeze(0)
                depth_colormap = colormap_image(batch["depth"])
                depth_colormap[:, mask] = 255.0
                depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
                image = np.hstack((rgb, depth_colormap))
                cv2.namedWindow("RGB-D".format(), cv2.WINDOW_AUTOSIZE)
                cv2.imshow("RGB-D".format(), image)
                key = cv2.waitKey(1)

            # First frame mapping
            # *******建立初始的地图和位姿估计********
            if i == 0:
                self.first_frame_mapping(
                    batch, self.config["mapping"]["first_iters"]
                )  # first_iters=1000

            # 建立每一帧的地图和位姿估计
            # Tracking + Mapping
            else:
                # ***************** Tracking*****************
                if self.config["tracking"]["iter_point"] > 0:  # 代码中是 False
                    # *****通过点云来跟踪当前帧的相机位姿********
                    self.tracking_pc(batch, i)
                # 使用当前的 rgb 损失,深度损失,sdf 损失来跟踪当前帧的相机位姿
                self.tracking_render(batch, i)
                # ***************** Mapping*****************
                if i % self.config["mapping"]["map_every"] == 0:
                    self.current_frame_mapping(batch, i)
                    # ***************** Global BA*****************
                    self.global_BA(batch, i)

                # Add keyframe

                if self.config["active"]["isActive"]:
                    H = self.dataset.H
                    W = self.dataset.W
                    focal_x = self.config["cam"]["fx"]
                    focal_y = self.config["cam"]["fy"]
                    poses = torch.stack(self.dataset.poses).to(self.device)  # 直接在CPU上合并

                    if i % self.config["active"]["active_iter"] == 0:
                        # *********添加关键帧***************************
                        print("\nstart evaluation:")
                        print("get rays")
                        # TODO pose 后期替换为程序获得
                        rays = np.stack(
                            [get_rays_np(H, W, focal_x, focal_y, p) for p in poses.cpu().numpy()[:, :3, :4]],
                            0)  # [N, ro+rd, H, W, 3]

                        print("done, concats")

                        # get all holdout rays (candidate)
                        # images 应该是 138*800*800*3

                        rays_rgb_all = torch.cat([torch.tensor(rays).to(self.device), all_images[:, None]],
                                                 1)  # [N, ro+rd+rgb, H, W, 3]
                        rays_rgb_all = rays_rgb_all.permute(0, 2, 3, 1, 4)  # [N, H, W, ro+rd+rgb, 3]

                        rays_rgb_holdout = torch.cat(
                            [rays_rgb_all[j, ::self.downsample_rate, ::self.downsample_rate] for j in i_holdout], 0)
                        rays_rgb_holdout = torch.reshape(rays_rgb_holdout, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
                        rays_rgb_holdout = torch.transpose(rays_rgb_holdout, 0, 1)
                        batch_rays = rays_rgb_holdout[:2]
                        print('before evaluation:', i_train, i_holdout)
                        # capture new rays
                        self.model.eval()
                        indice = self.select_samples(
                            # 图像 H*W 368*496, samples=2048
                            # indice就是从 0 到 182528 之间随机选取 2048 个数作为 index
                            self.dataset.H, self.dataset.W,
                            self.config["mapping"]["sample"],  # 2048
                        )
                        # 每个像素在图像上的位置索引
                        # ******************************
                        indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
                        rays_d_cam = (
                            batch["direction"].squeeze(0)[indice_h, indice_w, :].to(self.device)
                        )
                        target_s = batch["rgb"].squeeze(0)[indice_h, indice_w, :].to(self.device)
                        target_d = (
                            batch["depth"].squeeze(0)[indice_h, indice_w]
                            .to(self.device).unsqueeze(-1)
                        )
                        c2w = batch["c2w"][0].to(self.device)

                        rays_o = c2w[None, :3, -1].repeat(self.config["mapping"]["sample"], 1)
                        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

                        self.model.forward(rays_o, rays_d, target_rgb, target_d)
                        # ******************************
                        hold_out_index = choose_new_k(H // self.downsample_rate, W // self.downsample_rate, focal_x,
                                                      focal_y, batch_rays, self.config["active"]["choose_k"])
                        i_train = np.append(i_train, i_holdout[hold_out_index])
                        i_holdout = np.delete(i_holdout, hold_out_index)
                        print('after evaluation:', i_train, i_holdout, hold_out_index)

                        # update training rays
                        rays_rgb_train = torch.stack([rays_rgb_all[i] for i in i_train], 0)  # train images only
                        rays_rgb_train = torch.reshape(rays_rgb_train, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]

                        print('shuffle rays')
                        np.random.shuffle(rays_rgb_train)

                        # **************************
                        self.keyframeDatabase.add_keyframe(
                            batch, filter_depth=self.config["mapping"]["filter_depth"]
                        )
                        print("add keyframe:", i)
                    # if (i % self.config["mapping"]["keyframe_every"] == 0):  # keyframe_every=5
                    #     self.keyframeDatabase.add_keyframe(
                    #         batch, filter_depth=self.config["mapping"]["filter_depth"]  # False
                    #     )
                    #     print("add keyframe:", i)

                if i % self.config["mesh"]["vis"] == 0:
                    self.save_mesh(i, voxel_size=self.config["mesh"]["voxel_eval"])
                    pose_relative = self.convert_relative_pose()
                    pose_evaluation(
                        self.pose_gt,
                        self.est_c2w_data,
                        1,
                        os.path.join(
                            self.config["data"]["output"],
                            self.config["data"]["exp_name"],
                        ),
                        i,
                    )
                    pose_evaluation(
                        self.pose_gt,
                        pose_relative,
                        1,
                        os.path.join(
                            self.config["data"]["output"],
                            self.config["data"]["exp_name"],
                        ),
                        i,
                        img="pose_r",
                        name="output_relative.txt",
                    )

                    if cfg["mesh"]["visualisation"]:
                        cv2.namedWindow("Traj:".format(), cv2.WINDOW_AUTOSIZE)
                        traj_image = cv2.imread(
                            os.path.join(
                                self.config["data"]["output"],
                                self.config["data"]["exp_name"],
                                "pose_r_{}.png".format(i),
                            )
                        )
                        # best_traj_image = cv2.imread(os.path.join(best_logdir_scene, "pose_r_{}.png".format(i)))
                        # image_show = np.hstack((traj_image, best_traj_image))
                        image_show = traj_image
                        cv2.imshow("Traj:".format(), image_show)
                        key = cv2.waitKey(1)

        model_savepath = os.path.join(
            self.config["data"]["output"],
            self.config["data"]["exp_name"],
            "checkpoint{}.pt".format(i),
        )

        self.save_ckpt(model_savepath)
        self.save_mesh(i, voxel_size=self.config["mesh"]["voxel_final"])

        pose_relative = self.convert_relative_pose()
        pose_evaluation(
            self.pose_gt,
            self.est_c2w_data,
            1,
            os.path.join(
                self.config["data"]["output"], self.config["data"]["exp_name"]
            ),
            i,
        )
        pose_evaluation(
            self.pose_gt,
            pose_relative,
            1,
            os.path.join(
                self.config["data"]["output"], self.config["data"]["exp_name"]
            ),
            i,
            img="pose_r",
            name="output_relative.txt",
        )

        # TODO: Evaluation of reconstruction


if __name__ == "__main__":

    print("Start running...")
    parser = argparse.ArgumentParser(
        description="Arguments for running the NICE-SLAM/iMAP*."
    )
    parser.add_argument("--config", type=str, help="Path to config file.")
    parser.add_argument(
        "--input_folder",
        type=str,
        help="input folder, this have higher priority, can overwrite the one in config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="output folder, this have higher priority, can overwrite the one in config file",
    )

    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg["data"]["output"] = args.output

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg["data"]["exp_name"])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy("coslam.py", os.path.join(save_path, "coslam.py"))

    with open(os.path.join(save_path, "config.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg, indent=4))
    start_time = time.time()  # 开始时间

    slam = CoSLAM(cfg)

    slam.run()
    # print(slam.keyframeDatabase.frame_ids)
