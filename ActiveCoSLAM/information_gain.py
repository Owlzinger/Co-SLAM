import torch
import numpy as np


def choose_new_k(H, W, focal_x, focal_y, batch_rays, k, rays_o, rays_d, target_rgb, target_d):
    pres = []
    posts = []
    N = H * W
    n = batch_rays.shape[1] // N
    for i in range(n):
        with torch.no_grad():
            _, _, _, uncert, _, extras = render(
                H, W, focal_x, focal_y,
                chunk=None,
                rays=batch_rays[:, i * N: i * N + N, :],
                # [2,6400000,3]--> [2,160000,3]
                verbose=True, retraw=True,
            )
        # extras["raw"]: 160000,192,5
        # uncert[160000,],像素数量
        # uncert_render是渲染后的不确定度
        # uncert_pts是采样点的方差
        uncert_render = (
                uncert.reshape(-1, H * W, 1) + 1e-9
        )  # 1,160000,1 新数据集r2(holdout数据集)的不确定度 beta, \beta^2(r_2)
        # -1 表示自动计算该维度, 第二维度固定是 160000, 第三维度是 1, 第一维度自动计算
        uncert_pts = (
                extras["raw"][..., -1].reshape(
                    -1, H * W, N_samples + N_importance
                )
                + 1e-9
        )  # 1,160000,192 \beta^2(r_2(t_k)), 其实就是 160000*192的一个 2 维矩阵
        # extras["raw"][..., -1]:
        # extras["raw"]: 是一个五维矩阵 R G B A var(方差)
        # extras["raw"][..., -1]=extras["raw"][..., 4]: 取矩阵的最后一维度 即var(方差)
        weight_pts = extras["weights"].reshape(-1, H * W, N_samples + N_importance  # 128+64
                                               )  # 1,160000,192

        pre = uncert_pts.sum([1, 2])  # 1,160000,192->1
        # pre[i] 的值等于 uncert_pts[i, :, :].sum()，即 uncert_pts[i, :, :] 中所有元素的和。
        # weight_pts * weight_pts: 1,160000,192
        # weight_pts * weight_pts / uncert_render: 1,160000,192
        post = (1.0 / (1.0 / uncert_pts + weight_pts * weight_pts / uncert_render)).sum(
            [1, 2]
        )  # 1,160000,192->21760060
        pres.append(pre)
        posts.append(post)

    pres = torch.cat(pres, 0)  # 40,
    posts = torch.cat(posts, 0)  # 40
    index = torch.topk(pres - posts, k)[1].cpu().numpy()

    return index


def img2mse_uncert_alpha(x, y, uncert, alpha, w):
    return torch.mean((1 / (2 * (uncert + 1e-9).unsqueeze(-1))) * ((x - y) ** 2)) + 0.5 * torch.mean(
        torch.log(uncert + 1e-9)) + w * alpha.mean() + 4.0


def get_rays_np(H, W, focal_x, focal_y, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5) / focal_x, -(j - H * .5) / focal_y, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                    -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


def render(H, W, focal, chunk=1024 * 32, rays=None, c2w=None, ndc=True, near=0.0, far=1.0, use_viewdirs=False,
           c2w_staticcam=None, **kwargs, ):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1.0, rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(
        rays_d[..., :1]
    )
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ["rgb_map", "disp_map", "acc_map", "uncert_map", "alpha_map"]
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W - 1, W),
                          torch.linspace(0, H - 1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i - W * .5) / focal, -(j - H * .5) / focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3],
                       -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d
