import copy
import glob
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from imageio import imread
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm
from path import Path
import random

from datasets.utils import get_camera_rays, alphanum_key, as_intrinsics_matrix


class BaseDataset(Dataset):
    def __init__(self, cfg):
        self.png_depth_scale = cfg["cam"]["png_depth_scale"]
        self.H, self.W = (
            cfg["cam"]["H"] // cfg["data"]["downsample"],
            cfg["cam"]["W"] // cfg["data"]["downsample"],
        )

        self.fx, self.fy = (
            cfg["cam"]["fx"] // cfg["data"]["downsample"],
            cfg["cam"]["fy"] // cfg["data"]["downsample"],
        )
        self.cx, self.cy = (
            cfg["cam"]["cx"] // cfg["data"]["downsample"],
            cfg["cam"]["cy"] // cfg["data"]["downsample"],
        )
        self.distortion = (
            np.array(cfg["cam"]["distortion"]) if "distortion" in cfg["cam"] else None
        )
        self.crop_size = cfg["cam"]["crop_edge"] if "crop_edge" in cfg["cam"] else 0
        self.ignore_w = cfg["tracking"]["ignore_edge_W"]
        self.ignore_h = cfg["tracking"]["ignore_edge_H"]

        self.total_pixels = (self.H - self.crop_size * 2) * (
            self.W - self.crop_size * 2
        )
        self.num_rays_to_save = int(
            self.total_pixels * cfg["mapping"]["n_pixels"]
        )  # n_pixels: 0.05 # num of pixels saved for each frame
        #  cfg['mapping']['n_pixels']: 0.05

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()


class KITTIDataset(BaseDataset):
    def __init__(
        self,
        cfg,
        basedir,
        trainskip=1,
        downsample_factor=1,
        translation=0.0,
        sc_factor=1.0,
        crop=0,
    ):
        super(KITTIDataset, self).__init__(cfg)

        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop
        self.img_files = sorted(
            glob.glob(os.path.join(self.basedir, "sequneces/07", "*.png")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )
        self.depth_paths = sorted(
            glob.glob(os.path.join(self.basedir, "depth/07", "*.png")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )
        self.load_poses(os.path.join(self.basedir, "poses", "07.txt"))

        # self.depth_cleaner = cv2.rgbd.DepthCleaner_create(cv2.CV_32F, 5)

        self.rays_d = None
        self.frame_ids = range(0, len(self.img_files))
        self.num_frames = len(self.frame_ids)

        if self.config["cam"]["crop_edge"] > 0:
            self.H -= self.config["cam"]["crop_edge"] * 2
            self.W -= self.config["cam"]["crop_edge"] * 2
            self.cx -= self.config["cam"]["crop_edge"]
            self.cy -= self.config["cam"]["crop_edge"]

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        color_path = self.img_files[index]
        depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        if ".png" in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif ".exr" in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            raise NotImplementedError()

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.0
        depth_data = (
            depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor
        )

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            self.fx = self.fx // self.downsample_factor
            self.fy = self.fy // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        edge = self.config["cam"]["crop_edge"]
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]

        if self.rays_d is None:
            self.rays_d = get_camera_rays(
                self.H, self.W, self.fx, self.fy, self.cx, self.cy
            )

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w": self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d,
        }

        return ret

    def load_poses(self, path):
        self.poses = []
        pose_paths = sorted(
            glob.glob(os.path.join(path, "*.txt")),
            key=lambda x: int(os.path.basename(x)[:-4]),
        )
        for pose_path in pose_paths:
            with open(pose_path, "r") as f:
                lines = f.readlines()
            ls = []
            for line in lines:
                l = list(map(float, line.split(" ")))
                ls.append(l)
            c2w = np.array(ls).reshape(4, 4)
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            self.poses.append(c2w)


def main():
    # 写一个测试函数,加载地址在/home/aneins/Codes/dataset/KITTI/odometry/dataset的数据
    cfg = {
        "dataset": "tum",
        "data": {
            "downsample": 1,
            "sc_factor": 1,
            "translation": 0,
            "num_workers": 4,
            "datadir": "/home/aneins/Codes/dataset/TUM/rgbd_dataset_freiburg1_desk",
            "trainskip": 1,
            "output": "output/TUM/fr_desk",
            "exp_name": "ac_demo",
        },
        "mapping": {
            "sample": 2048,
            "first_mesh": True,
            "iters": 20,
            "cur_frame_iters": 0,
            "lr_embed": 0.01,
            "lr_decoder": 0.01,
            "lr_rot": 0.001,
            "lr_trans": 0.001,
            "keyframe_every": 5,
            "map_every": 5,
            "n_pixels": 0.05,
            "first_iters": 1000,
            "optim_cur": True,
            "min_pixels_cur": 100,
            "map_accum_step": 1,
            "pose_accum_step": 5,
            "map_wait_step": 0,
            "filter_depth": False,
            "bound": [[-3.5, 3], [-3, 3], [-3, 3]],
            "marching_cubes_bound": [[-3.5, 3], [-3, 3], [-3, 3]],
        },
        "tracking": {
            "iter": 10,
            "sample": 1024,
            "pc_samples": 40960,
            "lr_rot": 0.01,
            "lr_trans": 0.01,
            "ignore_edge_W": 20,
            "ignore_edge_H": 20,
            "iter_point": 0,
            "wait_iters": 100,
            "const_speed": True,
            "best": False,
        },
        "grid": {
            "enc": "HashGrid",
            "tcnn_encoding": True,
            "hash_size": 16,
            "voxel_color": 0.04,
            "voxel_sdf": 0.02,
            "oneGrid": True,
        },
        "pos": {"enc": "OneBlob", "n_bins": 16},
        "decoder": {
            "geo_feat_dim": 15,
            "hidden_dim": 32,
            "num_layers": 2,
            "num_layers_color": 2,
            "hidden_dim_color": 32,
            "tcnn_network": False,
        },
        "cam": {
            "H": 480,
            "W": 640,
            "fx": 517.3,
            "fy": 516.5,
            "cx": 318.6,
            "cy": 255.3,
            "crop_edge": 8,
            "crop_size": [384, 512],
            "distortion": [0.2624, -0.9531, -0.0054, 0.0026, 1.1633],
            "near": 0,
            "far": 6,
            "depth_trunc": 5.0,
            "png_depth_scale": 5000.0,
        },
        "training": {
            "rgb_weight": 1.0,
            "depth_weight": 0.1,
            "sdf_weight": 5000,
            "fs_weight": 10,
            "eikonal_weight": 0,
            "smooth_weight": 1e-08,
            "smooth_pts": 64,
            "smooth_vox": 0.04,
            "smooth_margin": 0.0,
            "n_samples_d": 64,
            "range_d": 0.25,
            "n_range_d": 21,
            "n_importance": 0,
            "perturb": 1,
            "white_bkgd": False,
            "trunc": 0.05,
            "rot_rep": "axis_angle",
            "rgb_missing": 1.0,
        },
        "mesh": {
            "resolution": 512,
            "render_color": False,
            "vis": 500,
            "voxel_eval": 0.05,
            "voxel_final": 0.03,
            "visualisation": False,
        },
        "active": {
            "isActive": True,
            "active_iter": 10,
            "init_image": 10,
            "choose_k": 10,
            "w": 0.01,
            "downsample_rate": 2,
            "beta_min": 0.01,
        },
    }
    basedir = "/home/aneins/Codes/dataset/KITTI/odometry/dataset"
    dataset = KITTIDataset(cfg, basedir)
    # 调用函数并输出第二个摄像机的内参矩阵
    camera_index = 2  #  左彩色相机 P2
    intrinsics_matrix = dataset.extract_camera_intrinsics(
        "path/to/sequence/01/calib.txt", camera_index
    )
    print("Intrinsic matrix K for the second camera:")
    print(intrinsics_matrix)


main()
