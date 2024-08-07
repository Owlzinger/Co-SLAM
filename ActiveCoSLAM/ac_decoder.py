# Package imports
import torch
import torch.nn as nn
import tinycudann as tcnn


class ColorNet(nn.Module):
    def __init__(self, config, input_ch=4, geo_feat_dim=16,
                 hidden_dim_color=64, num_layers_color=3):
        super(ColorNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim_color = hidden_dim_color
        self.num_layers_color = num_layers_color

        self.model = self.get_model(config['decoder']['tcnn_network'])

    def forward(self, input_feat):
        # h = torch.cat([embedded_dirs, geo_feat], dim=-1)
        return self.model(input_feat)

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('Color net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch + self.geo_feat_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_color,
                    "n_hidden_layers": self.num_layers_color - 1,
                },
                # dtype=torch.float
            )

        color_net = []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = self.input_ch + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color

            if l == self.num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = self.hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers_color - 1:
                color_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(color_net))


class SDFNet(nn.Module):
    def __init__(self, config, input_ch=3, geo_feat_dim=16, hidden_dim=64, num_layers=2):
        super(SDFNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = self.geo_feat_dim
        self.model = self.get_model(tcnn_network=config['decoder']['tcnn_network'])

    def forward(self, x, return_geo=True):
        out = self.model(x)

        if return_geo:  # return feature
            return out
        else:
            return out[..., :1]

    def get_model(self, tcnn_network=False):
        if tcnn_network:
            print('SDF net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch,
                # n_output_dims=1 + self.geo_feat_dim,
                n_output_dims=self.geo_feat_dim,  # 不需要加一因为第一个网络全部都是输出
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers - 1,
                },
                # dtype=torch.float
            )
        else:
            sdf_net = []
            for l in range(self.num_layers):  # num_layers=2
                if l == 0:
                    in_dim = self.input_ch
                else:
                    in_dim = self.hidden_dim

                if l == self.num_layers - 1:
                    # out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color ---> 16
                    out_dim = self.geo_feat_dim  # 不需要加一了, 因为第一个网络全部都需要输出
                else:
                    out_dim = self.hidden_dim

                sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if l != self.num_layers - 1:
                    sdf_net.append(nn.ReLU(inplace=True))

            return nn.Sequential(*nn.ModuleList(sdf_net))


class ColorSDFNet(nn.Module):
    '''
    Color grid + SDF grid
    # 允许颜色网络直接访问原始的颜色嵌入信息(ColorSDFNet的特点)

    '''

    def __init__(self, config, input_ch=3, input_ch_pos=12):
        raise NotImplementedError

    #     super(ColorSDFNet, self).__init__()
    #     self.config = config
    #     self.color_net = ColorNet(config,
    #                               input_ch=input_ch + input_ch_pos,
    #                               geo_feat_dim=config['decoder']['geo_feat_dim'],
    #                               hidden_dim_color=config['decoder']['hidden_dim_color'],
    #                               num_layers_color=config['decoder']['num_layers_color'])
    #     self.sdf_net = SDFNet(config,
    #                           input_ch=input_ch + input_ch_pos,
    #                           geo_feat_dim=config['decoder']['geo_feat_dim'],
    #                           hidden_dim=config['decoder']['hidden_dim'],
    #                           num_layers=config['decoder']['num_layers'])
    #
    # def forward(self, embed, embed_pos, embed_color):
    #
    #     if embed_pos is not None:
    #         h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True)
    #     else:
    #         h = self.sdf_net(embed, return_geo=True)
    #
    #     sdf, geo_feat = h[..., :1], h[..., 1:]
    #     if embed_pos is not None:
    #         rgb = self.color_net(torch.cat([embed_pos, embed_color, geo_feat], dim=-1))
    #     else:
    #         rgb = self.color_net(torch.cat([embed_color, geo_feat], dim=-1))
    #
    #     return torch.cat([rgb, sdf], -1)


class ColorSDFNet_v3(nn.Module):
    '''
    No color grid
    颜色信息不能直接访问,而是要结合空间编码

    '''

    def __init__(self, config, input_ch=3, input_ch_pos=12):
        super(ColorSDFNet_v3, self).__init__()
        self.config = config
        self.color_net = ColorNet(config,
                                  input_ch=input_ch_pos,
                                  geo_feat_dim=config['decoder']['geo_feat_dim'],
                                  hidden_dim_color=config['decoder']['hidden_dim_color'],
                                  num_layers_color=config['decoder']['num_layers_color'])
        self.sdf_net = SDFNet(config,
                              input_ch=input_ch + input_ch_pos,
                              geo_feat_dim=config['decoder']['geo_feat_dim'],
                              hidden_dim=config['decoder']['hidden_dim'],
                              num_layers=config['decoder']['num_layers'])
        self.beta_min = 1e-4
        self.softplus = nn.Softplus()
        W = self.sdf_net.output_dim
        self.feature_linear = nn.Linear(W, W)
        self.sdf_linear = nn.Linear(W, 1)
        self.uncertainty_linear = nn.Linear(W, 1)

    def forward(self, embed, embed_pos):
        # embed_pos: [174080,48]
        # embed: [174080,32]

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True)
        else:
            h = self.sdf_net(embed, return_geo=True)

        sdf = self.sdf_linear(h)
        beta = self.softplus(self.uncertainty_linear(h)) + self.beta_min
        geo_feat = self.feature_linear(h)
        if embed_pos is not None:
            rgb = self.color_net(torch.cat([embed_pos, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([geo_feat], dim=-1))

        return torch.cat([rgb, sdf, beta], -1)


if __name__ == '__main__':
    import torch

    config = {
        'dataset': 'tum',
        'data': {'downsample': 1, 'sc_factor': 1, 'translation': 0, 'num_workers': 4,
                 'datadir': 'data/TUM/rgbd_dataset_freiburg1_desk', 'trainskip': 1,
                 'output': 'output/TUM/fr_desk', 'exp_name': 'demo'},
        'mapping': {'sample': 2048, 'first_mesh': True, 'iters': 20, 'cur_frame_iters': 0, 'lr_embed': 0.01,
                    'lr_decoder': 0.01, 'lr_rot': 0.001, 'lr_trans': 0.001, 'keyframe_every': 5, 'map_every': 5,
                    'n_pixels': 0.05, 'first_iters': 1000, 'optim_cur': True, 'min_pixels_cur': 100,
                    'map_accum_step': 1,
                    'pose_accum_step': 5, 'map_wait_step': 0, 'filter_depth': False,
                    'bound': [[-3.5, 3], [-3, 3], [-3, 3]], 'marching_cubes_bound': [[-3.5, 3], [-3, 3], [-3, 3]]},
        'tracking': {'iter': 10, 'sample': 1024, 'pc_samples': 40960, 'lr_rot': 0.01, 'lr_trans': 0.01,
                     'ignore_edge_W': 20, 'ignore_edge_H': 20, 'iter_point': 0, 'wait_iters': 100, 'const_speed': True,
                     'best': False},
        'grid': {'enc': 'HashGrid', 'tcnn_encoding': True, 'hash_size': 16, 'voxel_color': 0.04, 'voxel_sdf': 0.02,
                 'oneGrid': True}, 'pos': {'enc': 'OneBlob', 'n_bins': 16},
        'decoder': {'geo_feat_dim': 16, 'hidden_dim': 32, 'num_layers': 2, 'num_layers_color': 2,
                    'hidden_dim_color': 32,
                    'tcnn_network': False},
        'cam': {'H': 480, 'W': 640, 'fx': 517.3, 'fy': 516.5, 'cx': 318.6, 'cy': 255.3, 'png_depth_scale': 5000.0,
                'crop_edge': 8, 'near': 0, 'far': 5, 'depth_trunc': 5.0, 'crop_size': [384, 512],
                'distortion': [0.2624, -0.9531, -0.0054, 0.0026, 1.1633]},
        'training': {'rgb_weight': 1.0, 'depth_weight': 0.1, 'sdf_weight': 5000, 'fs_weight': 10, 'eikonal_weight': 0,
                     'smooth_weight': 1e-08, 'smooth_pts': 64, 'smooth_vox': 0.04, 'smooth_margin': 0.0,
                     'n_samples_d': 64,
                     'range_d': 0.25, 'n_range_d': 21, 'n_importance': 0, 'perturb': 1, 'white_bkgd': False,
                     'trunc': 0.05,
                     'rot_rep': 'axis_angle', 'rgb_missing': 1.0},
        'mesh': {'resolution': 512, 'render_color': False, 'vis': 500, 'voxel_eval': 0.05, 'voxel_final': 0.03,
                 'visualisation': False}, 'inherit_from': 'configs/Tum/tum.yaml'}
    model = ColorSDFNet_v3(config, input_ch=32, input_ch_pos=48)
    # embed_pos: [174080,48]
    # embed: [174080,32]
    embed = torch.randn(174080, 32)
    embed_pos = torch.randn(174080, 48)
    model.forward(embed, embed_pos)
    # 导出模型为ONNX格式
    torch.onnx.export(model,  # 导出的模型
                      (embed, embed_pos),  # 模型输入 (元组或张量列表)
                      "color_sdf_netv2.onnx",  # 保存的文件路径
                      input_names=['embed', 'embed_pos'],  # 输入名称
                      output_names=['output'],
                      export_params=True,
                      )
    colornet = ColorNet(config,
                        input_ch=48,
                        geo_feat_dim=config['decoder']['geo_feat_dim'],
                        hidden_dim_color=config['decoder']['hidden_dim_color'],
                        num_layers_color=config['decoder']['num_layers_color'])
    inputfeat4colornet = torch.randn(174080, 63)

    torch.onnx.export(colornet,  # 导出的模型
                      inputfeat4colornet,  # 模型输入 (元组或张量列表)
                      "colornet.onnx",  # 保存的文件路径
                      input_names=['inputfeat'],  # 输入名称
                      output_names=['output'],
                      export_params=True,
                      )
    sdfnet = SDFNet(config,
                    input_ch=32 + 48,
                    geo_feat_dim=config['decoder']['geo_feat_dim'],
                    hidden_dim=config['decoder']['hidden_dim'],
                    num_layers=config['decoder']['num_layers'])
    x4SDFNet = torch.randn(174080, 32 + 48)
    torch.onnx.export(sdfnet,  # 导出的模型
                      x4SDFNet,  # 模型输入 (元组或张量列表)
                      "sdfnet.onnx",  # 保存的文件路径
                      input_names=['x'],  # 输入名称
                      output_names=['output'],
                      export_params=True,
                      )
