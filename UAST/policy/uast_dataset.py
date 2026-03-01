import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from ruamel.yaml import YAML
import time
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split

def generate_sphere_pointcloud(center, radius, num_points=1000):
    """生成球体点云"""
    points = []
    for _ in range(num_points):
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        x = center[0] + radius * np.sin(theta) * np.cos(phi)
        y = center[1] + radius * np.sin(theta) * np.sin(phi)
        z = center[2] + radius * np.cos(theta)
        points.append([x, y, z])
    return np.array(points)

# 添加生成四旋翼无人机点云的函数
def generate_quadcopter_pointcloud(center, num_points=2000):
    """生成四旋翼无人机点云模型"""
    points = []
    
    # 无人机结构参数
    body_size = 0.2  # 机身尺寸
    arm_length = 0.5  # 机臂长度
    propeller_radius = 0.15  # 螺旋桨半径
    
    # 1. 生成机身点云 (立方体)
    body_points = int(num_points * 0.15)  # 15%点用于机身
    for _ in range(body_points):
        x = center[0] + np.random.uniform(-body_size/2, body_size/2)
        y = center[1] + np.random.uniform(-body_size/2, body_size/2)
        z = center[2] + np.random.uniform(-body_size/2, body_size/2)
        points.append([x, y, z])
    
    # 2. 生成机臂点云 (四个方向的长方体)
    arm_points = int(num_points * 0.3)  # 30%点用于机臂
    arm_points_per_arm = arm_points // 4
    
    # 前、后、左、右四个机臂方向
    arm_directions = [
        (1, 0),  # 前 (x轴正方向)
        (-1, 0), # 后 (x轴负方向)
        (0, 1),  # 右 (y轴正方向)
        (0, -1)  # 左 (y轴负方向)
    ]
    
    for dx, dy in arm_directions:
        for _ in range(arm_points_per_arm):
            # 机臂上的点
            arm_pos = np.random.uniform(0, arm_length)
            x = center[0] + dx * (body_size/2 + arm_pos)
            y = center[1] + dy * (body_size/2 + arm_pos)
            z = center[2] + np.random.uniform(-0.05, 0.05)  # 机臂厚度
            points.append([x, y, z])
    
    # 3. 生成螺旋桨点云 (四个螺旋桨)
    prop_points = int(num_points * 0.55)  # 55%点用于螺旋桨
    prop_points_per_prop = prop_points // 4
    
    for i, (dx, dy) in enumerate(arm_directions):
        # 螺旋桨中心位置
        prop_center_x = center[0] + dx * (body_size/2 + arm_length)
        prop_center_y = center[1] + dy * (body_size/2 + arm_length)
        prop_center_z = center[2]
        
        for _ in range(prop_points_per_prop):
            # 生成螺旋桨圆盘上的点 (极坐标)
            theta = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(0, propeller_radius)
            x = prop_center_x + r * np.cos(theta)
            y = prop_center_y + r * np.sin(theta)
            z = prop_center_z + np.random.uniform(-0.02, 0.02)  # 螺旋桨厚度
            points.append([x, y, z])
    
    return np.array(points)

# 添加生成人物模型点云的函数
def generate_human_pointcloud(center, num_points=3000):
    """生成人物模型点云模型，center表示人物头顶的位置"""
    points = []

    temp_dis = 1
    
    # 人物结构参数 (假设平均身高1.7m)
    height = 1.7
    head_radius = 0.12  # 头部半径
    torso_width = 0.4   # 躯干宽度
    torso_depth = 0.2   # 躯干深度
    arm_length = 0.4    # 手臂长度
    arm_radius = 0.04   # 手臂粗细
    leg_length = 0.8    # 腿部长度
    leg_radius = 0.05    # 腿部粗细
    
    # 计算各部位中心位置 (相对于头顶center)
    head_center = np.array([center[0], center[1], center[2] - head_radius + temp_dis])
    torso_center = np.array([center[0], center[1], center[2] - 2*head_radius - height*0.2 + temp_dis])
    
    # 1. 生成头部点云 (球体)
    head_points = int(num_points * 0.1)
    for _ in range(head_points):
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(0, np.pi)
        r = np.random.uniform(0, head_radius)
        x = head_center[0] + r * np.sin(theta) * np.cos(phi)
        y = head_center[1] + r * np.sin(theta) * np.sin(phi)
        z = head_center[2] + r * np.cos(theta)
        points.append([x, y, z])
    
    # 2. 生成躯干点云 (长方体)
    torso_points = int(num_points * 0.3)
    for _ in range(torso_points):
        x = torso_center[0] + np.random.uniform(-torso_width/2, torso_width/2)
        y = torso_center[1] + np.random.uniform(-torso_depth/2, torso_depth/2)
        z = torso_center[2] + np.random.uniform(-height*0.3, height*0.1)
        points.append([x, y, z])
    
    # 3. 生成手臂点云 (左右手臂)
    arm_points = int(num_points * 0.15 * 2)  # 15%每只手臂
    arm_points_per_arm = arm_points // 2
    
    # 左侧手臂
    left_arm_center = np.array([torso_center[0] - torso_width/2 - arm_radius,
                               torso_center[1],
                               torso_center[2] + height*0.05])
    for _ in range(arm_points_per_arm):
        # 手臂圆柱体
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, arm_radius)
        arm_pos = np.random.uniform(0, arm_length)
        x = left_arm_center[0] + r * np.cos(theta)
        y = left_arm_center[1] + r * np.sin(theta)
        z = left_arm_center[2] - arm_pos
        points.append([x, y, z])
    
    # 右侧手臂
    right_arm_center = np.array([torso_center[0] + torso_width/2 + arm_radius,
                                torso_center[1],
                                torso_center[2] - height*0.1])
    for _ in range(arm_points_per_arm):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, arm_radius)
        arm_pos = np.random.uniform(0, arm_length)
        x = right_arm_center[0] + r * np.cos(theta)
        y = right_arm_center[1] + r * np.sin(theta)
        z = right_arm_center[2] - arm_pos
        points.append([x, y, z])
    
    # 4. 生成腿部点云 (左右腿)
    leg_points = int(num_points * 0.2 * 2)  # 20%每条腿
    leg_points_per_leg = leg_points // 2
    
    # 左腿
    left_leg_center = np.array([torso_center[0] - torso_width*0.2,
                              torso_center[1]- torso_width*0.2,
                              torso_center[2] - height*0.8])
    for _ in range(leg_points_per_leg):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, leg_radius)
        leg_pos = np.random.uniform(0, leg_length)
        x = left_leg_center[0] + r * np.cos(theta)
        y = left_leg_center[1] + r * np.sin(theta)
        z = left_leg_center[2] + leg_pos
        points.append([x, y, z])
    
    # 右腿
    right_leg_center = np.array([torso_center[0] + torso_width*0.2,
                               torso_center[1]+ torso_width*0.2,
                               torso_center[2] - height*0.8])
    for _ in range(leg_points_per_leg):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, leg_radius)
        leg_pos = np.random.uniform(0, leg_length)
        x = right_leg_center[0] + r * np.cos(theta)
        y = right_leg_center[1] + r * np.sin(theta)
        z = right_leg_center[2] + leg_pos
        points.append([x, y, z])
    
    return np.array(points)

class UASTDataset(Dataset):
    def __init__(self, mode='train', val_ratio=0.1):
        super(UASTDataset, self).__init__()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg = YAML().load(open(os.path.join(base_dir, "../config/traj_opt.yaml"), 'r'))
        # image params
        self.height = int(cfg["image_height"])
        self.width = int(cfg["image_width"])
        self.mask_height = int(cfg["mask_height"])
        self.mask_width = int(cfg["mask_width"])
        # ramdom state: x-direction: log-normal distribution, yz-direction: normal distribution
        scale = cfg["velocity"] / cfg["vel_align"]
        self.vel_scale = scale * cfg["vel_align"]
        self.acc_scale = scale * scale * cfg["acc_align"]
        self.vx_lognorm_mean = np.log(1 - cfg["vx_mean_unit"])
        self.vx_logmorm_sigma = np.log(cfg["vx_std_unit"])
        self.v_mean = np.array([cfg["vx_mean_unit"], cfg["vy_mean_unit"], cfg["vz_mean_unit"]])
        self.v_std = np.array([cfg["vx_std_unit"], cfg["vy_std_unit"], cfg["vz_std_unit"]])
        self.a_mean = np.array([cfg["ax_mean_unit"], cfg["ay_mean_unit"], cfg["az_mean_unit"]])
        self.a_std = np.array([cfg["ax_std_unit"], cfg["ay_std_unit"], cfg["az_std_unit"]])
        self.goal_length = 2.0 * cfg['radio_range']
        self.goal_pitch_std = cfg["goal_pitch_std"]
        self.goal_yaw_std = cfg["goal_yaw_std"]
        if mode == 'train': self.print_data()

        # 添加相机内参（用于投影）
        self.fx = cfg["camera"]["fx"]
        self.fy = cfg["camera"]["fy"]
        self.cx = cfg["camera"]["cx"]
        self.cy = cfg["camera"]["cy"]
        self.max_depth_dist = cfg["camera"]["max_depth_dist"]  # 添加最大深度值

        # 添加目标点云参数（复用实时代码逻辑）
        # self.target_relative_points = generate_sphere_pointcloud([0, 0, 0], 0.3, num_points=1000)  # 相对坐标点云（中心在原点）
        self.target_relative_points = generate_quadcopter_pointcloud([0, 0, 0], num_points=2000)  # 相对坐标点云（中心在原点）
        # self.target_relative_points = generate_human_pointcloud([0, 0, 0], num_points=3000)  # 相对坐标点云（中心在原点）

        # dataset
        print("Loading", mode, "dataset, it may take a while...")
        data_dir = os.path.join(base_dir, "../", cfg["dataset_path"])
        self.img_list, self.map_idx, self.positions, self.quaternions = [], [], np.empty((0, 3), dtype=np.float32), np.empty((0, 4), dtype=np.float32)
        self.rgb_list = []
        datafolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
        datafolders.sort(key=lambda x: int(os.path.basename(x)))
        print("Datafolders:")
        for folder in datafolders:
            print("    ", folder)

        for data_idx in range(len(datafolders)):
            datafolder = datafolders[data_idx]

            image_file_names = [filename
                                for filename in os.listdir(datafolder)
                                if os.path.splitext(filename)[1] == '.png'and filename.startswith('img_')]
            image_file_names.sort(key=lambda x: int(x.split('.')[0].split("_")[1]))  # sort by filename to align with the label

            rgb_file_names = [filename for filename in os.listdir(datafolder) 
                             if os.path.splitext(filename)[1] == '.png' and filename.startswith('rgb_')]
            rgb_file_names.sort(key=lambda x: int(x.split('.')[0].split("_")[1]))

            states = np.loadtxt(data_dir + f"/pose-{data_idx}.csv", delimiter=',', skiprows=1).astype(np.float32)
            positions = states[:, 0:3]
            quaternions = states[:, 3:7]

            # 修改：同时划分深度图和RGB图的训练/验证集
            depth_train, depth_val, rgb_train, rgb_val, positions_train, positions_val, quaternions_train, quaternions_val = train_test_split(
                image_file_names, rgb_file_names, positions, quaternions, test_size=val_ratio, random_state=0)

            if mode == 'train':
                # 加载深度图（保持原逻辑）
                images = [cv2.imread(os.path.join(datafolder, f), -1).astype(np.float32) for f in depth_train]
                # 加载RGB图（新增）
                rgbs = [cv2.imread(os.path.join(datafolder, f), cv2.IMREAD_COLOR).astype(np.float32) for f in rgb_train]
                
                self.img_list.extend(images)
                self.rgb_list.extend(rgbs)
                self.positions = np.vstack((self.positions, positions_train.astype(np.float32)))
                self.quaternions = np.vstack((self.quaternions, quaternions_train.astype(np.float32)))
            elif mode == 'valid':
                images = [cv2.imread(os.path.join(datafolder, f), -1).astype(np.float32) for f in depth_val]
                rgbs = [cv2.imread(os.path.join(datafolder, f), cv2.IMREAD_COLOR).astype(np.float32) for f in rgb_val]
                
                self.img_list.extend(images)
                self.rgb_list.extend(rgbs)
                self.positions = np.vstack((self.positions, positions_val.astype(np.float32)))
                self.quaternions = np.vstack((self.quaternions, quaternions_val.astype(np.float32)))
            else:
                raise ValueError(f"Invalid mode {mode}. Choose from 'train', 'valid'.")

            self.map_idx.extend([data_idx] * len(images))

        # NOTE: The depth images are normalized from 0–20m to a 0–1 and converted to int16 during data collection.
        self.img_list = [np.expand_dims(
                        cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_NEAREST) / 65535.0,
                        axis=0)
                        for img in self.img_list]
        
        # RGB图预处理（新增）
        self.rgb_list = [cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_NEAREST) 
                        for img in self.rgb_list]
        self.rgb_list = [img.transpose(2, 0, 1) / 255.0 for img in self.rgb_list]  # 转为CHW格式并归一化


        print(f"=============== {mode.capitalize()} Data Summary ===============")
        print(f"{'Images'      :<12} | Count: {len(self.img_list):<3} |  Shape: {self.img_list[0].shape}")
        print(f"{'Positions'   :<12} | Count: {self.positions.shape[0]:<3} |  Shape: {self.positions.shape[1]}")
        print(f"{'Quaternions' :<12} | Count: {self.quaternions.shape[0]:<3} |  Shape: {self.quaternions.shape[1]}")
        print("==================================================")
        print(mode.capitalize(), "data loaded!")

        # 初始化可见性统计计数器
        self.total_goals = 0
        self.visible_goals = 0

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        vel_b, acc_b = self._get_random_state()

        # generate random goal in front of the quadrotor.
        q_wxyz = self.quaternions[item, :]  # q: wxyz
        R_WB = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        euler_angles = R_WB.as_euler('ZYX', degrees=False)  # [yaw(z) pitch(y) roll(x)]
        R_wB = R.from_euler('ZYX', [0, euler_angles[1], euler_angles[2]], degrees=False)
        while True:
            goal_w = self._get_random_goal()    # 只是世界坐标系的旋转，没有平移，相当于是在原点的goal
            goal_b = R_wB.inv().apply(goal_w)
            # 这里误导性太大了，就是先在一个机身坐标不考虑旋转情况下在大前方上下一定范围生成随机目标
            # 然后再忽略yaw旋转为机身本地，让大部分目标点无论无人机如何yaw角转都在无人机朝向前方范围内
            # pitch 要放在wB中是确保目标点在世界坐标系下也是在这个范围内

            # 1. 生成目标点云（世界坐标系）：相对坐标点云 + 目标世界坐标
            target_points_world = self.target_relative_points + goal_w  # 目标点云世界坐标

            # 2. 坐标转换：世界坐标系 -> 相机坐标系
            R_WC = R_WB.as_matrix()  # 世界到相机的旋转矩阵（假设相机与机体坐标系一致）
            R_CW = R_WC.T  # 相机到世界的旋转矩阵
            target_points_camera = np.dot(R_CW, (target_points_world).T).T  # 目标点相机坐标 = R_CW*(目标世界坐标 - 相机世界坐标)

            # 3. 过滤相机前方的点（x > 0，相机坐标系x轴向前）
            valid_indices = target_points_camera[:, 0] > 0
            target_points_camera = target_points_camera[valid_indices]

            # 4. 投影到图像平面 (u, v)
            if len(target_points_camera) == 0:
                if self.visible_goals / (self.total_goals+1e-6) < 0.95:
                    continue  # 如果当前可见比例过低，重新生成目标
                # 无有效点，直接返回原图
                depth_img = self.img_list[item]
                rgb_img = self.rgb_list[item]
                mask = np.zeros((self.mask_height, self.mask_width), dtype=np.float32)
            else:
                # 计算投影坐标
                u = (target_points_camera[:, 1] * self.fx / target_points_camera[:, 0]) + self.cx
                v = (target_points_camera[:, 2] * self.fy / target_points_camera[:, 0]) + self.cy
                u = self.width - 1 - u   # 翻转u坐标（图像宽度方向）
                v = self.height - 1 - v  # 翻转v坐标（图像高度方向）
                # 创建有效的坐标掩码（仅保留图像范围内的点）
                valid_mask = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
                # 应用掩码过滤所有相关数组
                u = u[valid_mask].astype(int)
                v = v[valid_mask].astype(int)
                target_points_camera = target_points_camera[valid_mask]

                # 5. 准备深度图和RGB图（复制避免修改原数据）
                depth_img = self.img_list[item].copy()  # shape: [1, H, W]
                rgb_img = self.rgb_list[item].copy()      # shape: [3, H, W]

                # 6. 计算可见性掩码（目标点深度 < 原深度图深度 或 原深度为NaN）
                visibility_mask = np.zeros(len(u), dtype=bool)
                for i in range(len(u)):
                    # 深度图值为归一化值 [0,1]，需转换为实际深度值比较
                    current_depth = depth_img[0, v[i], u[i]] * self.max_depth_dist
                    if np.isnan(current_depth) or current_depth > target_points_camera[i, 0]:
                        visibility_mask[i] = True
                        # 更新深度图（归一化到[0,1]）
                        depth_img[0, v[i], u[i]] = target_points_camera[i, 0] / self.max_depth_dist
                mask = np.zeros((self.height, self.width), dtype=np.float32)
                # 7. 在RGB图上绘制可见的目标点（红色，BGR格式归一化值）
                if np.any(visibility_mask):
                    visible_u = u[visibility_mask]
                    visible_v = v[visibility_mask]
                    rgb_img[:, visible_v, visible_u] = np.array([0, 0, 1])[:, np.newaxis]  # RGB图通道顺序为[C, H, W]，[0,0,1]对应红色
                    # 在掩码上标记可见区域
                    mask[visible_v, visible_u] = 1
                    self.visible_goals += 1
                else:
                    if self.visible_goals / (self.total_goals+1e-6) < 0.95:
                        continue  # 如果当前可见比例过低，重新生成目标
                mask = cv2.resize(mask, (self.mask_width, self.mask_height), interpolation=cv2.INTER_NEAREST)
            self.total_goals += 1
            break  # 有可见点或决定接受不可见目标，跳出循环

        # 计算当前可见性比例并打印
        visibility_ratio = self.visible_goals / self.total_goals if self.total_goals > 0 else 0
        # print(f"\rGoal Visibility: {self.visible_goals}/{self.total_goals} ({visibility_ratio*100:.2f}%)")
        random_obs = np.hstack((vel_b, acc_b, goal_b)).astype(np.float32)
        rot_wb = R_WB.as_matrix().astype(np.float32)  # transform to rot_matrix in numpy is faster than using quat in pytorch
        # vel & acc & goal are in body frame, NWU, and no-normalization

        mask = np.expand_dims(mask, axis=0)   
        return depth_img, rgb_img, mask, self.positions[item], rot_wb, random_obs, self.map_idx[item]

    def _get_random_state(self):
        while True:
            vel = self.vel_scale * (self.v_mean + self.v_std * np.random.randn(3))
            right_skewed_vx = -1
            while right_skewed_vx < 0:
                right_skewed_vx = self.vel_scale * np.random.lognormal(mean=self.vx_lognorm_mean, sigma=self.vx_logmorm_sigma, size=None)
                right_skewed_vx = -right_skewed_vx + 1.2 * self.vel_scale  # * 1.2 to ensure v_max can be sampled
            vel[0] = right_skewed_vx
            if np.linalg.norm(vel) < 1.2 * self.vel_scale:  # avoid outliers
                break

        while True:
            acc = self.acc_scale * (self.a_mean + self.a_std * np.random.randn(3))
            if np.linalg.norm(acc) < 1.2 * self.acc_scale:  # avoid outliers
                break
        return vel, acc

    def _get_random_goal(self):
        goal_pitch_angle = np.random.normal(0.0, self.goal_pitch_std)   
        goal_yaw_angle = np.random.normal(0.0, self.goal_yaw_std)   # 让90%的点落在这个[-2*std, 2*std]范围内
        goal_pitch_angle, goal_yaw_angle = np.radians(goal_pitch_angle), np.radians(goal_yaw_angle)
        # 球坐标系到直角坐标系的转换
        goal_w_dir = np.array([np.cos(goal_yaw_angle) * np.cos(goal_pitch_angle),
                               np.sin(goal_yaw_angle) * np.cos(goal_pitch_angle), np.sin(goal_pitch_angle)])
        # 10% probability to generate a nearby goal (× goal_length is actual length)
        # 小概率模拟近处的点，让无人机知道什么时候该停下来，这里实际上是再额外生成0.1概率的目标在[0,self.goal_length]范围内的目标
        # 如果我要执行的是追踪的话是不是就意味大多数目标就应该在附近
        # 这里我尝试了一下如果让0.5概率的目标在[0, self.goal_length]范围内的目标，直接导航训练效果很差
        random_near = np.random.rand()
        if random_near < 0.6:
            if random_near < 0.1:
                goal_w_dir = np.random.rand() * 0.5 * goal_w_dir   # 0.1概率 [0,0.5]]
            else:
                goal_w_dir = (1+np.random.rand())*0.5 * goal_w_dir  # 0.4 概率 [0.5,1.0]
        return self.goal_length * goal_w_dir

    def print_data(self):
        import scipy.stats as stats
        # 计算Vx 5% ~ 95% 区间
        p5 = self.vel_scale * np.exp(stats.norm.ppf(0.05, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))
        p95 = self.vel_scale * np.exp(stats.norm.ppf(0.95, loc=self.vx_lognorm_mean, scale=self.vx_logmorm_sigma))

        v_lower = self.vel_scale * (self.v_mean - 2 * self.v_std)
        v_upper = self.vel_scale * (self.v_mean + 2 * self.v_std)
        v_lower[0] = -p95 + 1.2 * self.vel_scale
        v_upper[0] = -p5 + 1.2 * self.vel_scale

        a_lower = self.acc_scale * (self.a_mean - 2 * self.a_std)
        a_upper = self.acc_scale * (self.a_mean + 2 * self.a_std)

        print("----------------- Sampling State --------------------")
        print("| X-Y-Z | Vel 90% Range(m/s)  | Acc 90% Range(m/s2) |")
        print("|-------|---------------------|---------------------|")
        for i in range(3):
            print(f"|  {i:^4} | {v_lower[i]:^9.1f}~{v_upper[i]:^9.1f} |"
                  f" {a_lower[i]:^9.1f}~{a_upper[i]:^9.1f} |")
        print("-----------------------------------------------------")
        print(f"| Goal Pitch 90% (deg)        | {-self.goal_pitch_std * 2:^9.1f}~{self.goal_pitch_std * 2:^9.1f} |")
        print(f"| Goal Yaw   90% (deg)        | {-self.goal_yaw_std * 2:^9.1f}~{self.goal_yaw_std * 2:^9.1f} |")
        print("-----------------------------------------------------")

    def plot_sample_distribution(self):
        import matplotlib.pyplot as plt
        # ===== 采样 =====
        N = 10000
        goals = np.array([self._get_random_goal() for _ in range(N)])
        states = np.array([self._get_random_state() for _ in range(N)])
        vels = np.stack([s[0] for s in states])
        accs = np.stack([s[1] for s in states])

        x, y, z = goals[:, 0], goals[:, 1], goals[:, 2]
        yaw = np.degrees(np.arctan2(y, x))  # 水平角 [-180, 180]
        pitch = np.degrees(np.arctan2(z, np.sqrt(x ** 2 + y ** 2)))  # 垂直角 [-90, 90]

        fig, axs = plt.subplots(3, 3, figsize=(15, 10))

        # Goal方向角分布
        axs[0, 0].hist(yaw, bins=180)
        axs[0, 0].set_title("Goal Yaw Distribution")
        axs[0, 0].set_xlabel("Yaw (deg)")
        axs[0, 0].set_xlim([-60, 60])
        axs[0, 0].grid(True)

        axs[0, 1].hist(pitch, bins=90)
        axs[0, 1].set_title("Goal Pitch Distribution")
        axs[0, 1].set_xlabel("Pitch (deg)")
        axs[0, 1].set_xlim([-60, 60])
        axs[0, 1].grid(True)

        # Goal往图像投影分布(未考虑机体旋转)
        axs[0, 2].scatter(yaw, pitch, s=2, alpha=0.3)
        axs[0, 2].set_title("Goal Distribution in Image")
        axs[0, 2].set_xlabel("Yaw (deg)")
        axs[0, 2].set_ylabel("Pitch (deg)")
        axs[0, 2].set_xlim([-45, 45])
        axs[0, 2].set_ylim([-30, 30])
        axs[0, 2].grid(True)

        # Velocity分布
        for i, name in enumerate(['Vx', 'Vy', 'Vz']):
            axs[1, i].hist(vels[:, i], bins=100)
            axs[1, i].set_title(f"Velocity {name}")
            axs[1, i].grid(True)

        # Acceleration分布
        for i, name in enumerate(['Ax', 'Ay', 'Az']):
            axs[2, i].hist(accs[:, i], bins=100)
            axs[2, i].set_title(f"Acceleration {name}")
            axs[2, i].grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    dataset = UASTDataset()
    dataset.plot_sample_distribution()
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    start = time.time()
    for epoch in range(1):
        last = time.time()
        for i, (depth, pos, quat, obs, id) in enumerate(data_loader):
            pass
    end = time.time()

    print("加载1个epoch总耗时：", end - start)
