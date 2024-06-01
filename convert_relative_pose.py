import torch


class KeyframeDatabase:
    def __init__(self, frame_ids):
        self.frame_ids = frame_ids


class PoseEstimator:
    def __init__(self, est_c2w_data, est_c2w_data_rel, keyframeDatabase, config):
        self.est_c2w_data = est_c2w_data
        self.est_c2w_data_rel = est_c2w_data_rel
        self.keyframeDatabase = keyframeDatabase
        self.config = config

    def convert_relative_pose(self):
        poses = {}
        keyframe_ids_list = list(self.keyframeDatabase.frame_ids)

        for i in range(len(self.est_c2w_data)):
            if i in keyframe_ids_list:
                poses[i] = self.est_c2w_data[i].float()  # 确保类型为 float
            else:
                # 找到最近的关键帧
                kf_id = max(kf for kf in keyframe_ids_list if kf < i)
                c2w_key = self.est_c2w_data[kf_id].float()  # 确保类型为 float
                delta = self.est_c2w_data_rel[i].float()  # 确保类型为 float
                poses[i] = delta @ c2w_key

        return poses


# 测试数据
est_c2w_data = {
    0: torch.eye(4),
    5: torch.tensor([[0.866, -0.5, 0, 0],
                     [0.5, 0.866, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
    10: torch.tensor([[0.866, 0.5, 0, 0],
                      [-0.5, 0.866, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]),
}

est_c2w_data_rel = {
    1: torch.tensor([[1, 0, 0, 1],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
    6: torch.tensor([[1, 0, 0, -1],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]]),
}

keyframeDatabase = KeyframeDatabase([0, 5, 10])
config = {
    "mapping": {
        "keyframe_every": 5
    }
}

pose_estimator = PoseEstimator(est_c2w_data, est_c2w_data_rel, keyframeDatabase, config)

# 生成相对位姿
converted_poses = pose_estimator.convert_relative_pose()

# 输出测试结果
for frame_id, pose in converted_poses.items():
    print(f"Frame {frame_id}:")
    print(pose)
