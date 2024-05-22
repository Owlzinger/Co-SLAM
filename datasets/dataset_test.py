import copy
import glob
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from datasets.utils import get_camera_rays, alphanum_key, as_intrinsics_matrix


def get_dataset(config):
    '''
    Get the dataset class from the config file.
    '''
    # if config['dataset'] == 'replica':
    #     dataset = ReplicaDataset

    # elif config['dataset'] == 'scannet':
    #     dataset = ScannetDataset
    #
    # elif config['dataset'] == 'synthetic':
    #     dataset = RGBDataset

    if config['dataset'] == 'tum':
        dataset = TUMDataset

    # elif config['dataset'] == 'azure':
    #     dataset = AzureDataset
    #
    # elif config['dataset'] == 'iphone':
    #     dataset = iPhoneDataset
    #
    # elif config['dataset'] == 'realsense':
    #     dataset = RealsenseDataset

    return dataset(config,
                   config['data']['datadir'],
                   trainskip=config['data']['trainskip'],
                   downsample_factor=config['data']['downsample'],
                   sc_factor=config['data']['sc_factor'])


class BaseDataset(Dataset):
    def __init__(self, cfg):
        self.png_depth_scale = cfg['cam']['png_depth_scale']
        self.H, self.W = cfg['cam']['H'] // cfg['data']['downsample'], \
                         cfg['cam']['W'] // cfg['data']['downsample']

        self.fx, self.fy = cfg['cam']['fx'] // cfg['data']['downsample'], \
                           cfg['cam']['fy'] // cfg['data']['downsample']
        self.cx, self.cy = cfg['cam']['cx'] // cfg['data']['downsample'], \
                           cfg['cam']['cy'] // cfg['data']['downsample']
        self.distortion = np.array(
            cfg['cam']['distortion']) if 'distortion' in cfg['cam'] else None
        self.crop_size = cfg['cam']['crop_edge'] if 'crop_edge' in cfg['cam'] else 0
        self.ignore_w = cfg['tracking']['ignore_edge_W']
        self.ignore_h = cfg['tracking']['ignore_edge_H']

        self.total_pixels = (self.H - self.crop_size * 2) * (self.W - self.crop_size * 2)
        self.num_rays_to_save = int(self.total_pixels * cfg['mapping']['n_pixels'])
        #  cfg['mapping']['n_pixels']: 0.05

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()


class TUMDataset(BaseDataset):
    def __init__(self, cfg, basedir, align=True, trainskip=1,
                 downsample_factor=1, translation=0.0,
                 sc_factor=1., crop=0, load=True):
        super(TUMDataset, self).__init__(cfg)

        self.config = cfg
        self.basedir = basedir
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop

        self.color_paths, self.depth_paths, self.poses = self.loadtum(
            basedir, frame_rate=32)

        self.frame_ids = range(0, len(self.color_paths))
        self.num_frames = len(self.frame_ids)

        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None

        self.rays_d = None
        sx = self.crop_size[1] / self.W
        sy = self.crop_size[0] / self.H
        self.fx = sx * self.fx
        self.fy = sy * self.fy
        self.cx = sx * self.cx
        self.cy = sy * self.cy
        self.W = self.crop_size[1]
        self.H = self.crop_size[0]

        if self.config['cam']['crop_edge'] > 0:
            self.H -= self.config['cam']['crop_edge'] * 2
            self.W -= self.config['cam']['crop_edge'] * 2
            self.cx -= self.config['cam']['crop_edge']
            self.cy -= self.config['cam']['crop_edge']

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses
        rgb图像深度图以及真是值 GT 使用不同的频率采集的,因此需要将他们对齐
        """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def loadtum(self, datapath, frame_rate=-1):
        """ read video data in tum-rgbd format """
        if os.path.isfile(os.path.join(datapath, 'groundtruth.txt')):
            pose_list = os.path.join(datapath, 'groundtruth.txt')
        elif os.path.isfile(os.path.join(datapath, 'pose.txt')):
            pose_list = os.path.join(datapath, 'pose.txt')

        image_list = os.path.join(datapath, 'rgb.txt')
        depth_list = os.path.join(datapath, 'depth.txt')

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 1:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(
            tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        images, poses, depths, intrinsics = [], [], [], []
        # images: 是图片的路径
        # rgb: 是图片的 rgb 信息
        # depths: 是深度图的路径
        inv_pose = None
        for ix in tqdm(indicies, desc='loading images depths poses'):
            (i, j, k) = associations[ix]
            images += [os.path.join(datapath, image_data[i, 1])]
            depths += [os.path.join(datapath, depth_data[j, 1])]

            c2w = self.pose_matrix_from_quaternion(pose_vecs[k])
            # if inv_pose is None:
            #     inv_pose = np.linalg.inv(c2w)
            #     c2w = np.eye(4)
            # else:
            #     c2w = inv_pose@c2w
            c2w[:3, 1] *= -1
            c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses += [c2w]

        return images, depths, poses

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):

        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.config['cam']['fx'],
                                      self.config['cam']['fy'],
                                      self.config['cam']['cx'],
                                      self.config['cam']['cy']])
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            self.fx = self.fx // self.downsample_factor
            self.fy = self.fy // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))

        if self.crop_size is not None:
            # follow the pre-processing step in lietorch, actually is resize
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        edge = self.config['cam']['crop_edge']
        if edge > 0:
            # crop image edge, there are invalid value on the edge of the color image
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w": self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d
        }
        return ret

    def slice(self, indices):
        """
        Returns a new TUMDataset instance containing only the elements specified by the indices list.

        Args:
            indices (list of int): List of indices to include in the sliced dataset.

        Returns:
            TUMDataset: A new dataset instance with the specified subset of data.
        """
        new_dataset = copy.copy(self)

        # Select the data subsets based on the provided indices
        new_dataset.color_paths = [self.color_paths[i] for i in indices]
        new_dataset.depth_paths = [self.depth_paths[i] for i in indices]
        new_dataset.poses = [self.poses[i] for i in indices]

        # Update the frame IDs and number of frames in the sliced dataset
        new_dataset.frame_ids = [self.frame_ids[i] for i in indices]
        new_dataset.num_frames = len(new_dataset.frame_ids)
        # Since the data is already loaded and sliced, no need to load again
        return new_dataset

    def __add__(self, other):
        # 检查other是否也是MyClass的实例
        if not isinstance(other, TUMDataset):
            return NotImplemented

        new_dataset = copy.copy(self)
        # Select the data subsets based on the provided indices
        new_dataset.color_paths = self.color_paths + other.color_paths
        new_dataset.depth_paths = self.depth_paths + other.depth_paths
        new_dataset.poses = self.poses + other.poses
        new_dataset.frame_ids = self.frame_ids + other.frame_ids
        new_dataset.num_frames = len(new_dataset.frame_ids)

        return new_dataset

    def slice_except(self, indices):
        """
        Returns a new TUMDataset instance containing all the elements except those specified by the indices list.
        """
        all_indices = set(range(self.num_frames))
        remaining_indices = list(all_indices - set(indices))
        return self.slice(remaining_indices)


class KITTIDataset2(BaseDataset):
    def __init__(self, cfg, basedir, sequence_number, trainskip=1,
                 downsample_factor=1, translation=0.0,
                 sc_factor=1., crop=0, load=True):
        super(KITTIDataset, self).__init__(cfg)

        self.config = cfg
        self.basedir = basedir
        self.sequence_number = sequence_number
        self.trainskip = trainskip
        self.downsample_factor = downsample_factor
        self.translation = translation
        self.sc_factor = sc_factor
        self.crop = crop

        self.color_paths, self.depth_paths, self.poses = self.load_kitti(
            basedir, sequence_number, frame_rate=32)

        self.frame_ids = range(0, len(self.color_paths))
        self.num_frames = len(self.frame_ids)

        self.crop_size = cfg['cam']['crop_size'] if 'crop_size' in cfg['cam'] else None

        self.rays_d = None
        sx = self.crop_size[1] / self.W
        sy = self.crop_size[0] / self.H
        self.fx = sx * self.fx
        self.fy = sy * self.fy
        self.cx = sx * self.cx
        self.cy = sy * self.cy
        self.W = self.crop_size[1]
        self.H = self.crop_size[0]

        if self.config['cam']['crop_edge'] > 0:
            self.H -= self.config['cam']['crop_edge'] * 2
            self.W -= self.config['cam']['crop_edge'] * 2
            self.cx -= self.config['cam']['crop_edge']
            self.cy -= self.config['cam']['crop_edge']

    def pose_matrix_from_quaternion(self, pvec):
        """ convert 4x4 pose matrix to (t, q) """
        from scipy.spatial.transform import Rotation

        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(pvec[3:]).as_matrix()
        pose[:3, 3] = pvec[:3]
        return pose

    def associate_frames(self, tstamp_image, tstamp_pose, max_dt=0.08):
        """ pair images and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            k = np.argmin(np.abs(tstamp_pose - t))
            if np.abs(tstamp_pose[k] - t) < max_dt:
                associations.append((i, k))

        return associations

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ', dtype=np.unicode_, skiprows=skiprows)
        return data

    def load_kitti(self, datapath, sequence_number, frame_rate=-1):
        """ read video data in KITTI format """
        pose_list = os.path.join(datapath, 'poses', f'{sequence_number:02d}.txt')
        image_list = os.path.join(datapath, 'sequences', f'{sequence_number:02d}', 'image_2')
        depth_list = os.path.join(datapath, 'sequences', f'{sequence_number:02d}',
                                  'image_3')  # Assumes depth images are stored similarly

        image_data = sorted(os.listdir(image_list))
        depth_data = sorted(os.listdir(depth_list))
        pose_data = np.loadtxt(pose_list).reshape(-1, 3, 4)

        tstamp_image = np.array([float(name.split('.')[0]) for name in image_data])
        tstamp_pose = np.arange(len(pose_data))
        associations = self.associate_frames(tstamp_image, tstamp_pose)

        indices = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indices[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indices += [i]

        images, poses, depths = [], [], []
        for ix in tqdm(indices, desc='loading images depths poses'):
            (i, k) = associations[ix]
            images.append(os.path.join(image_list, image_data[i]))
            depths.append(os.path.join(depth_list, depth_data[i]))

            pose = np.eye(4)
            pose[:3, :4] = pose_data[k]
            pose[:3, 1] *= -1
            pose[:3, 2] *= -1
            pose = torch.from_numpy(pose).float()
            poses.append(pose)

        return images, depths, poses

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]

        color_data = cv2.imread(color_path)
        if '.png' in depth_path:
            depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        elif '.exr' in depth_path:
            raise NotImplementedError()
        if self.distortion is not None:
            K = as_intrinsics_matrix([self.config['cam']['fx'],
                                      self.config['cam']['fy'],
                                      self.config['cam']['cx'],
                                      self.config['cam']['cy']])
            color_data = cv2.undistort(color_data, K, self.distortion)

        color_data = cv2.cvtColor(color_data, cv2.COLOR_BGR2RGB)
        color_data = color_data / 255.
        depth_data = depth_data.astype(np.float32) / self.png_depth_scale * self.sc_factor

        H, W = depth_data.shape
        color_data = cv2.resize(color_data, (W, H))

        if self.downsample_factor > 1:
            H = H // self.downsample_factor
            W = W // self.downsample_factor
            self.fx = self.fx // self.downsample_factor
            self.fy = self.fy // self.downsample_factor
            color_data = cv2.resize(color_data, (W, H), interpolation=cv2.INTER_AREA)
            depth_data = cv2.resize(depth_data, (W, H), interpolation=cv2.INTER_NEAREST)

        if self.rays_d is None:
            self.rays_d = get_camera_rays(self.H, self.W, self.fx, self.fy, self.cx, self.cy)

        color_data = torch.from_numpy(color_data.astype(np.float32))
        depth_data = torch.from_numpy(depth_data.astype(np.float32))

        if self.crop_size is not None:
            color_data = color_data.permute(2, 0, 1)
            color_data = F.interpolate(
                color_data[None], self.crop_size, mode='bilinear', align_corners=True)[0]
            depth_data = F.interpolate(
                depth_data[None, None], self.crop_size, mode='nearest')[0, 0]
            color_data = color_data.permute(1, 2, 0).contiguous()

        edge = self.config['cam']['crop_edge']
        if edge > 0:
            color_data = color_data[edge:-edge, edge:-edge]
            depth_data = depth_data[edge:-edge, edge:-edge]

        ret = {
            "frame_id": self.frame_ids[index],
            "c2w": self.poses[index],
            "rgb": color_data,
            "depth": depth_data,
            "direction": self.rays_d
        }
        return ret

    def slice(self, indices):
        new_dataset = copy.copy(self)
        new_dataset.color_paths = [self.color_paths[i] for i in indices]
        new_dataset.depth_paths = [self.depth_paths[i] for i in indices]
        new_dataset.poses = [self.poses[i] for i in indices]
        new_dataset.frame_ids = [self.frame_ids[i] for i in indices]
        new_dataset.num_frames = len(new_dataset.frame_ids)
        return new_dataset

    def __add__(self, other):
        if not isinstance(other, KITTIDataset):
            return NotImplemented

        new_dataset = copy.copy(self)
        new_dataset.color_paths = self.color_paths + other.color_paths
        new_dataset.depth_paths = self.depth_paths + other.depth_paths
        new_dataset.poses = self.poses + other.poses
        new_dataset.frame_ids = self.frame_ids + other.frame_ids
        new_dataset.num_frames = len(new_dataset.frame_ids)
        return new_dataset

    def slice_except(self, indices):
        all_indices = set(range(self.num_frames))
        remaining_indices = list(all_indices - set(indices))
        return self.slice(remaining_indices)


import os
import numpy as np
import pykitti
import torch
from torch.utils.data import Dataset
from PIL import Image


class KITTIDataset(Dataset):
    def __init__(self, base_dir, sequence, calib_dir=None):
        """
        Args:
            base_dir (str): Path to the base directory of the KITTI dataset.
            sequence (str): Sequence id to load (e.g., '00').
            calib_dir (str, optional): Path to the calibration directory.
        """
        self.dataset = pykitti.odometry(base_dir, sequence, frames=None)
        if calib_dir is not None:
            self.dataset.load_calib(calib_dir)
        self.sequence = sequence

    def __len__(self):
        return len(self.dataset.poses)

    def __getitem__(self, index):
        frame_id = index
        pose = self.dataset.poses[index]
        rgb = self._load_image(index)
        rays_direction = self._compute_rays_direction(pose)

        return {
            'frame_id': frame_id,
            'c2w': pose,
            'rgb': rgb,
            'depth': None,  # No depth data available
            'direction': rays_direction
        }

    def _load_image(self, index):
        """
        Helper function to load images from the dataset.
        Args:
            index (int): Index of the image to load.
        Returns:
            torch.Tensor: Loaded image as a tensor.
        """
        img_file = os.path.join(self.dataset.sequence_path, 'image_2', '{:06d}.png'.format(index))
        img = Image.open(img_file)
        img = img.convert('RGB')
        img = np.array(img, dtype=np.float32)
        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]

        return img

    def _compute_rays_direction(self, pose):
        """
        Compute rays direction for the given pose.
        Args:
            pose (np.ndarray): 4x4 pose matrix.
        Returns:
            torch.Tensor: Computed rays direction.
        """
        rays_direction = pose[:3, :3].dot(np.array([0, 0, 1]))  # Assuming the ray direction is along z-axis
        return torch.from_numpy(rays_direction).float()


# Example usage:
# dataset = KITTIDataset(base_dir='/path/to/kitti/dataset', sequence='00')
# data = dataset[0]
# print(data)

if __name__ == '__main__':
    # main function
    cfg = {'dataset': 'tum',
           'data': {'downsample': 1, 'sc_factor': 1, 'translation': 0, 'num_workers': 4,
                    'datadir': 'data/TUM/rgbd_dataset_freiburg1_desk', 'trainskip': 1, 'output': 'output/TUM/fr_desk',
                    'exp_name': 'demo'},
           'mapping': {'sample': 2048, 'first_mesh': True, 'iters': 20, 'cur_frame_iters': 0, 'lr_embed': 0.01,
                       'lr_decoder': 0.01, 'lr_rot': 0.001, 'lr_trans': 0.001, 'keyframe_every': 5, 'map_every': 5,
                       'n_pixels': 0.05, 'first_iters': 1000, 'optim_cur': True, 'min_pixels_cur': 100,
                       'map_accum_step': 1,
                       'pose_accum_step': 5, 'map_wait_step': 0, 'filter_depth': False,
                       'bound': [[-3.5, 3], [-3, 3], [-3, 3]], 'marching_cubes_bound': [[-3.5, 3], [-3, 3], [-3, 3]]},
           'tracking': {'iter': 10, 'sample': 1024, 'pc_samples': 40960, 'lr_rot': 0.01, 'lr_trans': 0.01,
                        'ignore_edge_W': 20, 'ignore_edge_H': 20, 'iter_point': 0, 'wait_iters': 100,
                        'const_speed': True,
                        'best': False},
           'grid': {'enc': 'HashGrid', 'tcnn_encoding': True, 'hash_size': 16, 'voxel_color': 0.04, 'voxel_sdf': 0.02,
                    'oneGrid': True}, 'pos': {'enc': 'OneBlob', 'n_bins': 16},
           'decoder': {'geo_feat_dim': 15, 'hidden_dim': 32, 'num_layers': 2, 'num_layers_color': 2,
                       'hidden_dim_color': 32,
                       'tcnn_network': False},
           'cam': {'H': 480, 'W': 640, 'fx': 517.3, 'fy': 516.5, 'cx': 318.6, 'cy': 255.3, 'crop_edge': 8,
                   'crop_size': [384, 512], 'distortion': [0.2624, -0.9531, -0.0054, 0.0026, 1.1633], 'near': 0,
                   'far': 6,
                   'depth_trunc': 5.0, 'png_depth_scale': 5000.0},
           'training': {'rgb_weight': 1.0, 'depth_weight': 0.1, 'sdf_weight': 5000, 'fs_weight': 10,
                        'eikonal_weight': 0,
                        'smooth_weight': 1e-08, 'smooth_pts': 64, 'smooth_vox': 0.04, 'smooth_margin': 0.0,
                        'n_samples_d': 64, 'range_d': 0.25, 'n_range_d': 21, 'n_importance': 0, 'perturb': 1,
                        'white_bkgd': False, 'trunc': 0.05, 'rot_rep': 'axis_angle', 'rgb_missing': 1.0},
           'mesh': {'resolution': 512, 'render_color': False, 'vis': 10, 'voxel_eval': 0.05, 'voxel_final': 0.03,
                    'visualisation': False},
           'active': {'isActive': True, 'active_iter': 10, 'init_image': 10, 'choose_k': 10, 'w': 0.01,
                      'downsample_rate': 2, 'beta_min': 0.01}}
    # data2 = TUMDataset(cfg, basedir="/home/aneins/Codes/Co-SLAM/data/TUM/rgbd_dataset_freiburg1_desk")
    dataset = KITTIDataset("/home/aneins/Codes/dataset/KITTI/odometry/", "03")
    data = dataset[0]
    train = data.slice(np.arange(0, 5))
    holdout = data.slice(np.arange(5, 10))
    c = holdout + train
    hold_out_index = [0, 1, 2, 3]

    print("the top info gain Frame-ids: ", [holdout.frame_ids[i] for i in hold_out_index])

    newholdout = holdout.slice_except(hold_out_index)

    print(data)
