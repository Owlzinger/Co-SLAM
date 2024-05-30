import torch
from torch.utils.data import DataLoader


def get_topK_info_gain_index(self, i):
    downsampled_H, downsampled_W = (
        self.dataset.H // 2, self.dataset.W // 2,
    )  # 只取 1/4 的点,
    # 原论文是 1/2的点, 但会出现 OOM
    samples_num = downsampled_H * downsampled_W  # 采样点数量
    # *********添加关键帧***************************
    # print('before evaluation:', i_train, i_holdout.min(), i_holdout.max())
    # *******************************************
    print("\nstart evaluation:")
    indice = self.select_samples(
        # 图像 H*W 368*496, samples=45632
        # indice就是从 0 到 182528 之间随机选取 45632 个数作为 index
        self.dataset.H, self.dataset.W, samples_num)
    indice_h, indice_w = (
        indice % self.dataset.H,
        indice // self.dataset.H,
    )
    pres = []
    posts = []
    self.model.eval()
    # 从第 i 帧开始后面的十个图像作为 holdout 数据集,计算信息增益
    # 第一次根据信息增益选择的时候, i=20, 关键帧库中的关键帧是 0,5,10,15.
    # 我想选择 16-25帧作为保留集, i-4=16, i-4+11=26
    holdout_dataset = self.dataset.slice(range(i - 4, i - 4 + 10))
    holdout_loader = DataLoader(holdout_dataset, num_workers=self.config["data"]["num_workers"])
    for i, batch in enumerate(holdout_loader):
        # tqdm库在 pycharm 终端有显示错误
        print("image:", i + 1, "/", len(holdout_dataset), "\r", end="")

        rays_d_cam = (batch["direction"].squeeze(0)[indice_h, indice_w, :].to(self.device))

        target_s = (batch["rgb"].squeeze(0)[indice_h, indice_w, :].to(self.device))

        target_d = (batch["depth"].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1))

        c2w = batch['c2w'][0].to(self.device)

        rays_o = c2w[None, :3, -1].repeat(samples_num, 1)
        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

        with torch.no_grad():
            rend_dict = self.model.forward(rays_o, rays_d, target_s, target_d)

        uncert_render = (rend_dict["uncert_map"].reshape(-1, samples_num, 1) + 1e-9)
        # 1,160000,1 新数据集r2(holdout数据集)的不确定度 beta, \beta^2(r_2)
        uncert_pts = (
                rend_dict["raw"][..., -1].reshape(
                    -1, samples_num,
                    self.config["training"]["n_samples_d"]
                    + self.config["training"]["n_importance"]
                    + self.config["training"]["n_range_d"],
                )
                + 1e-9
        )
        weight_pts = rend_dict["weights"].reshape(
            -1, samples_num, self.config["training"]["n_samples_d"]  # 64
                             + self.config["training"]["n_importance"]  # 0
                             + self.config["training"]["n_range_d"],  # 21
        )  # 1,160000,192

        pre = uncert_pts.sum([1, 2])  # 1,160000,192->1

        post = (1.0 / (1.0 / uncert_pts + weight_pts * weight_pts / uncert_render)).sum([1, 2])
        pres.append(pre)
        posts.append(post)

    pres = torch.cat(pres, 0)  # 40,
    posts = torch.cat(posts, 0)  # 40
    diff = pres - posts
    hold_out_index = (torch.topk(pres - posts, 1)[1].cpu().numpy())

    print(
        "the top info gain Frame-ids: ",
        [holdout_dataset.frame_ids[i] for i in hold_out_index],
    )
    return hold_out_index
