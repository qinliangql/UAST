import os
import glob
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from ruamel.yaml import YAML
from scipy.ndimage import distance_transform_edt


class TrackLoss(nn.Module):
    def __init__(self, L, sgm_time):
        super(TrackLoss, self).__init__()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg = YAML().load(open(os.path.join(base_dir, "../config/traj_opt.yaml"), 'r'))
        self.traj_num = cfg['horizon_num'] * cfg['vertical_num']
        self.map_expand_min = np.array(cfg['map_expand_min'])
        self.map_expand_max = np.array(cfg['map_expand_max'])
        self.d0 = cfg["d0"]
        self.r = cfg["r"]

        self._L = L
        self.sgm_time = sgm_time
        self.eval_points = 30
        self.device = self._L.device

        self.fov_angle = th.deg2rad(th.tensor(40.0))  # 假设FOV为±60度
        self.mid_ratio = [0.33, 0.66]  # 取轨迹中段
        # 从配置文件加载FOV参数（水平和垂直方向分开）
        self.horizon_fov = th.deg2rad(th.tensor(cfg["horizon_camera_fov"] / 2))  # 水平FOV/2 (弧度)
        self.vertical_fov = th.deg2rad(th.tensor(cfg["vertical_camera_fov"] / 2))  # 垂直FOV/2 (弧度)
        self.out_of_bounds_factor = 10.0  # 越界惩罚系数

        # SDF
        self.voxel_size = 0.2
        self.min_bounds = None  # shape: (N, 3)
        self.max_bounds = None  # shape: (N, 3)
        self.sdf_shapes = None  # shape: (1, 3)
        print("Building ESDF map...")
        data_dir = os.path.join(base_dir, "../", cfg["dataset_path"])
        self.sdf_maps = self.get_sdf_from_ply(data_dir)
        print("Map built!")

    def forward(self, Df, Dp, map_id, goal):
        """
        Args:
            Dp: decision parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            Df: fixed parameters: (batch_size, 3, 3) → [px, vx, ax; py, vy, ay; pz, vz, az]
            map_id: (batch_size) which esdf map to query
        Returns:
            cost_colli: (batch_size) → safety loss
        """
        batch_size = Dp.shape[0]
        L = self._L.unsqueeze(0).expand(batch_size, -1, -1)
        coe = self.get_coefficient_from_derivative(Dp, Df, L)

        track_cost = self.calculate_track_cost(coe, goal, map_id)

        return track_cost
    
    def calculate_track_cost(self, coe, goal, map_id):
        """计算轨迹中段对目标的观测质量"""
        # 采样中段时间点（1/3到2/3时间段）
        mid_points = int(self.eval_points * self.mid_ratio[0])
        mid_eval_points = int(self.eval_points * self.mid_ratio[1]) - mid_points
        t_mid = th.linspace(self.sgm_time*self.mid_ratio[0], 
                          self.sgm_time*self.mid_ratio[1],
                          mid_eval_points, device=self.device)
        
        # 获取中段轨迹点的位置和速度
        pos_mid = self.get_position_from_coeff(coe, t_mid)  # [B*H*V, self.eval_points/3, 3]
        pos_mid = pos_mid.reshape(-1, self.traj_num * pos_mid.shape[1], 3)    # [B, H*V*self.eval_points/3, 3]
        goal = goal.reshape(-1, self.traj_num, 3)   # [B, H*V, 3]
        goal = goal[:,0,:]
        
        # 计算视线遮挡损失（仅使用当前目标位置）
        blocked_mask = self.check_line_of_sight(pos_mid, goal.unsqueeze(1), map_id)  # [B, H*V*self.eval_points/3]
        occlusion_loss = blocked_mask.float().view(-1,self.eval_points//3).mean(dim=1)  # 遮挡惩罚系数 # [B, H*V]


        # 计算轨迹中段点的目标方向角与速度方向角
        vel_mid = self.get_velocity_from_coeff(coe, t_mid)  # [B, N, 3]
        vel_mid = vel_mid.reshape(-1, self.traj_num * vel_mid.shape[1], 3)    # [B, N*mid_eval_points, 3]

        # 1. 目标方向向量 (竖直分量)
        goal_dir = goal.unsqueeze(1) - pos_mid  # [B, N, 3]
        goal_dir_z = goal_dir[..., 2]  # 竖直方向分量
        goal_dist_xy = th.sqrt(goal_dir[..., 0]**2 + goal_dir[..., 1]**2 + 1e-6)  # 水平距离
        goal_pitch_angle = th.atan2(goal_dir_z, goal_dist_xy)  # 目标方向竖直角(与水平方向夹角)

        # 2. 速度方向向量 (竖直分量)
        vel_z = vel_mid[..., 2]  # 速度竖直分量
        vel_xy = th.sqrt(vel_mid[..., 0]**2 + vel_mid[..., 1]**2 + 1e-6)  # 水平速度
        vel_pitch_angle = th.atan2(vel_z, vel_xy)  # 速度方向竖直角

        # 3. 计算方向角差 (目标方向 - 速度方向)
        angle_diff = goal_pitch_angle - vel_pitch_angle

        # 4. 使用可微分的方法计算超出垂直FOV的程度
        # 计算超出边界的量，使用ReLU使其只在越界时产生惩罚
        pitch_deviation = th.abs(angle_diff) - self.vertical_fov
        # 使用ReLU创建可微分的惩罚，越界越多惩罚越大
        pitch_penalty = F.relu(pitch_deviation)  # [B, H*V*self.eval_points/3]
        
        # 计算最终俯仰角损失
        pitch_loss = pitch_penalty.view(-1, self.eval_points//3).mean(dim=1)  # [B]
        
        # 结合遮挡损失和FOV越界损失，使用合适的权重
        total_track_loss = occlusion_loss*4 + pitch_loss
        
        return total_track_loss*5.0
    
    def check_line_of_sight(self, start, end, map_id, num_samples=10):
        """检查轨迹点到目标的连线是否被障碍物遮挡
        Args:
            start: [B, H*V*self.eval_points/3, 3] 轨迹点位置
            end: [B, 3] 当前目标位置
            map_id: (B) 地图ID
        Returns:
            blocked: (B, N) 是否被遮挡
        """
        B, N = start.shape[:2]
        t = th.linspace(0, 1, num_samples, device=self.device)
        # 生成射线采样点
        line_points = start.unsqueeze(2) + t.view(1,1,-1,1) * (end - start).unsqueeze(2)
        line_points = line_points.reshape(B, -1, 3)  # [B, H*V*self.eval_points/3*num_samples, 3]
        
        # 查询所有采样点的SDF值
        dist = self.get_distance_cost(line_points, map_id)  # [B, H*V*self.eval_points/3*num_samples， 1]
        dist = dist.reshape(B, N, num_samples)  # [B, H*V*self.eval_points/3, num_samples]
        # 使用可微分的近似方法代替不可微分的判断
        # 计算距离小于阈值的程度，越接近阈值值越大
        occlusion_degree = F.relu(self.voxel_size - dist)  # 可微分
        # 聚合所有采样点的遮挡程度，而不是使用any()
        total_occlusion = occlusion_degree.mean(dim=2) # [B, H*V*self.eval_points/3]
    
        # 若任意采样点距离小于安全阈值，则判定为遮挡
        return total_occlusion 

    def get_distance_cost(self, pos, map_id):
        """
        pos:     (B, N, 3) - 点在世界坐标系下的位置
        map_id:  (B) - 每个 batch 使用哪张 sdf_map
        NOTE: Direct self.sdf_maps.expand(B, -1, -1, -1, -1) is the most memory-efficient and fastest, but only supports a single map.
              Using self.sdf_maps[map_id] results in significant memory usage and latency due to data copying.
              As a compromise, we adopt a map-cropping (get_batch_sdf) to support multiple maps.
        """
        B, N, _ = pos.shape

        # get local sdf maps
        sdf_maps, local_origin, local_shape = self.get_batch_sdf(pos, map_id)

        # 将 pos 转为 voxel 坐标：grid = (pos - min_bound) / voxel_size
        grid = (pos - local_origin.unsqueeze(1)) / self.voxel_size  # (B, N, 3)

        # 归一化 grid 到 [-1, 1]
        grid_point = 2.0 * grid / (local_shape - 1).unsqueeze(1) - 1.0  # (B, N, 3)

        grid_point = grid_point.view(B, 1, 1, N, 3)

        valid_mask = ((grid_point < 0.99).all(-1) & (grid_point > -0.99).all(-1)).squeeze(dim=1).squeeze(dim=1)  # (B, N)

        dist_query = F.grid_sample(sdf_maps, grid_point, mode='bilinear', padding_mode='zeros', align_corners=True)  # (B, 1, 1, 1, N)
        dist_query = dist_query.view(B, N)

        # Cost function
        cost = self.cost_function(dist_query)  # (B, N)

        cost = cost.masked_fill(~valid_mask, 0.0)

        return cost

    def cost_function(self, d):
        return th.exp(-(d - self.d0) / self.r)

    def get_coefficient_from_derivative(self, Dp, Df, L):
        coefficient = th.zeros(Dp.shape[0], 18, device=self.device)

        for i in range(3):
            d = th.cat([Df[:, i, :], Dp[:, i, :]], dim=1).unsqueeze(-1)  # [batch_size, num_dp + num_df, 1]
            coe = (L @ d).squeeze()   # [batch_size, 6]
            coefficient[:, 6 * i: 6 * (i + 1)] = coe

        return coefficient

    def get_position_from_coeff(self, coe, t):
        t_power = th.stack([th.ones_like(t), t, t ** 2, t ** 3, t ** 4, t ** 5], dim=-1).squeeze(-2)

        coe_x = coe[:, 0: 6]
        coe_y = coe[:, 6:12]
        coe_z = coe[:, 12:18]

        x = th.sum(t_power * coe_x.unsqueeze(1), dim=-1)
        y = th.sum(t_power * coe_y.unsqueeze(1), dim=-1)
        z = th.sum(t_power * coe_z.unsqueeze(1), dim=-1)

        pos = th.stack([x, y, z], dim=-1)
        return pos
    
    def get_velocity_from_coeff(self, coe, t):
        # 生成时间幂次：t^0, t^1, t^2, t^3, t^4
        t_power_velocity = th.stack([
            th.ones_like(t), 
            t, 
            t ** 2, 
            t ** 3, 
            t ** 4
        ], dim=-1).squeeze(-2)  # (batch_size, eval_points, 5)

        # 分解各轴系数并应用导数系数
        coe_x = coe[:, [1, 2, 3, 4, 5]] * th.tensor([1, 2, 3, 4, 5], device=self.device)  # [batch, 5]
        coe_y = coe[:, [7, 8, 9, 10, 11]] * th.tensor([1, 2, 3, 4, 5], device=self.device)
        coe_z = coe[:, [13, 14, 15, 16, 17]] * th.tensor([1, 2, 3, 4, 5], device=self.device)

        # 计算各轴速度分量
        vx = th.sum(t_power_velocity * coe_x.unsqueeze(1), dim=-1)  # (batch_size, eval_points)
        vy = th.sum(t_power_velocity * coe_y.unsqueeze(1), dim=-1)
        vz = th.sum(t_power_velocity * coe_z.unsqueeze(1), dim=-1)

        # 组合速度向量
        velocity = th.stack([vx, vy, vz], dim=-1)  # (batch_size, eval_points, 3)
        return velocity

    def get_batch_sdf(self, pos, map_id):
        """
            Crop all maps with the corresponding map_id in the batch to the same size and cover the pos.
        """
        min_bounds = self.min_bounds[map_id]  # [B, 3]
        sdf_shapes = self.sdf_shapes[map_id]  # [B, 3]

        min_pos = pos.amin(dim=1)  # [batch, 3]
        max_pos = pos.amax(dim=1)  # [batch, 3]
        min_indices = ((min_pos - min_bounds) / self.voxel_size).int()
        max_indices = ((max_pos - min_bounds) / self.voxel_size).int()
        spans = max_indices - min_indices  # [batch, 3]
        max_spans = spans.amax(dim=0)
        centers = (min_indices + max_indices) // 2  # [batch, 3]
        min_indices = centers - max_spans // 2 - 5  # [batch, 3]
        max_indices = centers + max_spans // 2 + 5  # [batch, 3]
        # Crop minimum value
        new_min_indices = min_indices.clamp(min=0)
        underflow_amount = new_min_indices - min_indices
        min_indices = new_min_indices
        max_indices = max_indices + underflow_amount

        # Crop maximum value
        new_max_indices = th.minimum(max_indices, sdf_shapes.int())
        overflow_amount = max_indices - new_max_indices
        max_indices = new_max_indices
        min_indices = min_indices - overflow_amount

        # Check for out-of-bounds indices. Although padding out-of-bound areas with zeros by F.pad() can prevent errors,
        # this situation rarely occurs, so for simplicity, we adjust min_indices directly.
        if (min_indices < 0).any():
            min_underflow = th.minimum(min_indices, th.zeros_like(min_indices))
            shift = (-min_underflow).max(dim=0).values
            min_indices = min_indices + shift

        sdf_maps = th.stack([self.sdf_maps[map_idx][0, :,
                             min_idx[2]:max_idx[2],
                             min_idx[1]:max_idx[1],
                             min_idx[0]:max_idx[0]]
                             for map_idx, min_idx, max_idx in zip(map_id.tolist(), min_indices.tolist(), max_indices.tolist())
                             ])
        local_origin = min_indices * self.voxel_size + min_bounds
        local_shape = max_indices - min_indices
        return sdf_maps, local_origin, local_shape

    def get_sdf_from_ply(self, path):
        sorted_files = self.read_sorted_ply_files(path)
        sdf_maps = []
        min_bounds, max_bounds, sdf_shapes = [], [], []

        # First pass to get all sdf_maps and record shape
        for file in sorted_files:
            pcd = o3d.io.read_point_cloud(file)
            min_bound = np.array(pcd.get_min_bound()) - self.map_expand_min
            max_bound = np.array(pcd.get_max_bound()) + self.map_expand_max
            points = np.asarray(pcd.points)
            print(f"    {os.path.basename(file)}: x=({min_bound[0] + self.map_expand_min[0]:.2f}, {max_bound[0] - self.map_expand_max[0]:.2f}), "
                  f"y=({min_bound[1] + self.map_expand_min[1]:.2f}, {max_bound[1] - self.map_expand_max[1]:.2f}), "
                  f"z=({min_bound[2] + self.map_expand_min[2]:.2f}, {max_bound[2] - self.map_expand_max[2]:.2f})")

            sdf_shape = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int)
            voxel_indices = ((points - min_bound) / self.voxel_size).astype(int)

            valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < sdf_shape), axis=1)
            voxel_indices = voxel_indices[valid_mask]

            occupancy = np.zeros(sdf_shape, dtype=np.uint8)
            occupancy[tuple(voxel_indices.T)] = 1

            obstacle_mask = occupancy == 1
            free_mask = occupancy == 0

            dist_to_obstacle = distance_transform_edt(free_mask) * self.voxel_size
            dist_inside_obstacle = distance_transform_edt(obstacle_mask) * self.voxel_size

            dist_to_obstacle[obstacle_mask] = -dist_inside_obstacle[obstacle_mask]

            sdf_tensor = th.from_numpy(dist_to_obstacle).float().unsqueeze(0).unsqueeze(0).permute(0, 1, 4, 3, 2).to(self.device)  # (1, 1, D, H, W)

            sdf_maps.append(sdf_tensor)
            sdf_shapes.append(sdf_tensor.shape[-3:][::-1])  # D, H, W -> X, Y, Z
            min_bounds.append(min_bound)
            max_bounds.append(max_bound)

        # Padding 所有 sdf_map 到最大尺寸, 以便堆积到batch并行处理
        # max_shape = np.max(np.stack(sdf_shapes), axis=0)
        # sdf_maps_padded = [self.pad_sdf_to_shape(sdf, max_shape) for sdf in sdf_maps]
        # sdf_maps_tensor = th.cat(sdf_maps, dim=0)  # shape: (N, 1, D, H, W)

        # maps shapes
        self.min_bounds = th.tensor(np.array(min_bounds), device=self.device).float()  # shape: (N, 3)
        self.max_bounds = th.tensor(np.array(max_bounds), device=self.device).float()  # shape: (N, 3)
        self.sdf_shapes = th.tensor(np.array(sdf_shapes), device=self.device).float()  # shape: (N, 3) order: (X, Y, Z)
        return sdf_maps  # shape: (N, 1, D, H, W)

    def read_sorted_ply_files(self, path):
        # 匹配所有以 pointcloud- 开头并以 .ply 结尾的文件, 并排序
        ply_files = glob.glob(os.path.join(path, 'pointcloud-*.ply'))

        def extract_index(filename):
            base = os.path.basename(filename)
            number_part = base.replace('pointcloud-', '').replace('.ply', '')
            return int(number_part)

        sorted_ply_files = sorted(ply_files, key=extract_index)

        return sorted_ply_files

    def pad_sdf_to_shape(self, sdf_map, target_shape):
        """
        Pads a 5D tensor (1, 1, D, H, W) to the target shape (D, H, W)
        """
        current_shape = sdf_map.shape[-3:]
        pad_sizes = [target - current for target, current in zip(target_shape[::-1], current_shape[::-1])]
        # Pad in (W, H, D) order, so reverse
        padding = [0, pad_sizes[0], 0, pad_sizes[1], 0, pad_sizes[2]]
        return F.pad(sdf_map, padding, mode='constant', value=0)