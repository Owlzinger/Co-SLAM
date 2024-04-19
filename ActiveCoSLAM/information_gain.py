import torch
import numpy as np


def quaternion2rotation_matrix(quaternion):
    """
    Convert a quaternion into a 3x3 rotation matrix.

    Parameters:
    quaternion (np.array): A numpy array with four elements [w, x, y, z],
                           where w is the scalar part, and x, y, z are the
                           components of the vector part.

    Returns:
    np.array: A 3x3 rotation matrix.
    """
    x, y, z, w = quaternion
    xx, xy, xz = x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z
    wx, wy, wz = w * x, w * y, w * z

    # Calculate the rotation matrix components
    matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])

    return matrix


def choose_new_k(H, W, focal, batch_rays, k, **render_kwargs_train):
    pres = []
    posts = []
    N = H * W
    n = batch_rays.shape[1] // N
    for i in range(n):
        with torch.no_grad():
            rgb, disp, acc, uncert, alpha, extras = render(
                H,
                W,
                focal,
                chunk=args.chunk,
                rays=batch_rays[:, i * N: i * N + N, :],
                # [2,6400000,3]--> [2,160000,3]
                verbose=True,
                retraw=True,
                **render_kwargs_train,
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
                    -1, H * W, args.N_samples + args.N_importance
                )
                + 1e-9
        )  # 1,160000,192 \beta^2(r_2(t_k)), 其实就是 160000*192的一个 2 维矩阵
        # extras["raw"][..., -1]:
        # extras["raw"]: 是一个五维矩阵 R G B A var(方差)
        # extras["raw"][..., -1]=extras["raw"][..., 4]: 取矩阵的最后一维度 即var(方差)
        weight_pts = extras["weights"].reshape(-1, H * W, args.N_samples + args.N_importance  # 128+64
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
