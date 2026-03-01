import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Point, PointStamped
from cv_bridge import CvBridge
from threading import Lock
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray


import cv2
import os
import time
import torch
import numpy as np
import argparse
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R

from control_msg import PositionCommand
from policy.uast_network import UASTNetworkTrack
from policy.poly_solver import *
from policy.state_transform import *

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")

# 添加生成球体点云的函数
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
def generate_quadcopter_pointcloud(center, num_points=6000):
    """生成四旋翼无人机点云模型"""
    points = []
    
    # 无人机结构参数
    body_size = 0.4  # 机身尺寸
    arm_length = 0.7  # 机臂长度
    propeller_radius = 0.2  # 螺旋桨半径
    
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

class UASTNet:
    def __init__(self, config, weight):
        self.config = config
        rospy.init_node('uast_net2', anonymous=False)
        # load params
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg = YAML().load(open(os.path.join(base_dir, "config/traj_opt.yaml"), 'r'))
        self.cfg = cfg
        self.height = cfg['image_height']
        self.width = cfg['image_width']
        self.mask_height = int(cfg["mask_height"])
        self.mask_width = int(cfg["mask_width"])
        self.min_dis, self.max_dis = 0.04, 20.0
        self.scale = {'435': 0.001, 'simulation': 1.0}.get(self.config['env'], 1.0)
        self.goal = np.array(self.config['goal'])
        self.plan_from_reference = self.config['plan_from_reference']
        self.use_trt = self.config['use_tensorrt']
        self.verbose = self.config['verbose']
        self.visualize = self.config['visualize']
        self.Rotation_bc = R.from_euler('ZYX', [0, self.config['pitch_angle_deg'], 0], degrees=True).as_matrix()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # variables
        self.bridge = CvBridge()
        self.odom = Odometry()
        self.odom_init = False
        self.last_yaw = 0.0
        self.ctrl_dt = 0.02
        self.ctrl_time = None
        self.desire_init = False
        self.arrive = False
        self.desire_pos = None
        self.desire_vel = None
        self.desire_acc = None
        self.optimal_poly_x = None
        self.optimal_poly_y = None
        self.optimal_poly_z = None
        self.lock = Lock()
        self.last_control_msg = None
        self.state_transform = StateTransform()
        self.lattice_primitive = LatticePrimitive.get_instance(cfg)
        self.traj_time = self.lattice_primitive.segment_time

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0
        self.pos = self.goal * 1.0   
        self.noisy_goal = self.goal * 1.0 

        # 添加目标球体参数
        self.target_sphere_radius = 0.3  # 球体半径
        self.target_sphere_points = None  # 球体点云
        self.target_sphere_center = self.goal  # 球体中心初始化为目标位置
        # 预生成球体相对坐标点（中心在原点），只生成一次
        # self.target_sphere_relative_points = generate_sphere_pointcloud([0, 0, 0], self.target_sphere_radius)
        self.target_sphere_relative_points = generate_quadcopter_pointcloud([0, 0, 0])
        # self.target_sphere_relative_points = generate_human_pointcloud([0, 0, 0])
        # 计算初始世界坐标点云（相对坐标 + 中心位置）
        self.target_sphere_points = self.target_sphere_relative_points + self.target_sphere_center

        # 添加RGB图像相关变量
        self.rgb_image = None
        self.sphere_uv = None  # 保存球体投影坐标 (u, v)
        self.sphere_uv_lock = Lock()  # 线程安全锁
        self.depth_share = None
        self.depth_mask = None
        self.depth_share_lock = Lock()

        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            state_dict = torch.load(weight, weights_only=True)
            self.policy = UASTNetworkTrack()
            self.policy.load_state_dict(state_dict)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
        self.warm_up()

        # ros publisher
        self.lattice_traj_pub = rospy.Publisher("/yopo_net2/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/yopo_net2/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/yopo_net2/trajs_visual", PointCloud2, queue_size=1)
        self.depth_vis_target_pub = rospy.Publisher("/yopo_net2/depth_vis_target", Image, queue_size=1)
        self.mask_vis_target_pub = rospy.Publisher("/yopo_net2/mask_vis_target", Image, queue_size=1)
        self.rgb_vis_target_pub = rospy.Publisher("/yopo_net2/rgb_vis_target", Image, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)
        # ros subscriber
        self.odom_sub = rospy.Subscriber(self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.config['depth_topic'], Image, self.callback_depth, queue_size=1)
        self.rgb_sub = rospy.Subscriber(self.config['rgb_topic'], Image, self.callback_rgb, queue_size=1)
        # self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal_point, queue_size=1)
        self.goal_sub = rospy.Subscriber(self.config['target_topic'], Odometry, self.callback_set_goal, queue_size=1)
        self.fov_pub = rospy.Publisher("/sensor_fov", PointCloud2, queue_size=1)
        # 添加目标点发布器
        self.goal_point_pub = rospy.Publisher("/yopo_net2/goal_point", PointStamped, queue_size=1)
        
        # 初始化可视化的FoV
        self.init_fov_visual()

        # 添加用于显示筛选后odom轨迹的publisher
        self.filtered_traj_pub = rospy.Publisher("/drone2/filtered_traj", PointCloud2, queue_size=1)
        self.trajectory_points = []  # 存储轨迹点的列表
        # 初始化上次发布的位置和标记
        self.last_published_pos = None
        self.filtered_odom_first_pub = False
        self.position_threshold = 0.5  # 距离阈值，单位：米

        # 初始化卡尔曼滤波器参数
        # self.initialize_kalman_filter()
        
        # 记录上次看到目标的时间
        self.last_seen_time = rospy.Time.now()
        self.is_target_visible = False

        # 2D地图参数，搜索时才需要
        self.map_resolution = 0.5  # 地图分辨率，单位：米/格
        self.map_size = 200  # 地图大小，单位：格，即200x200格的地图,100mx100m
        self.map_origin = np.array([-self.map_size/2 * self.map_resolution, -self.map_size/2 * self.map_resolution])  # 地图原点（左下）
        # 初始化2D voxel map，0表示未知，1表示障碍物
        self.voxel_map = np.zeros((self.map_size, self.map_size), dtype=np.uint8)
        # 用于可视化2D地图的publisher
        self.map_pub = rospy.Publisher("/drone2/2d_map", Image, queue_size=1)

        # 添加搜索相关参数
        self.search_mode = False  # 搜索模式标志
        self.last_search_time = rospy.Time.now()
        self.search_interval = 2.0  # 搜索目标更新间隔（秒）
        self.search_radius = 25.0  # 搜索半径（米）
        self.exploration_direction = 0  # 探索方向（角度）
        self.exploration_step = 45  # 每步探索旋转角度（度）

        self.search_count = 0
        self.track_count = 0
        self.dis_count = 0


        self.Rotation_wc = None

        # ros timer
        rospy.sleep(1.0)  # wait connection...
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        self.map_ctrl = rospy.Timer(rospy.Duration(0.1), self.publish_2d_map)
        rospy.Timer(rospy.Duration(0.1), self.pub_goal_point)
        print("UAST Net Node Ready!")
        rospy.spin()
    
    def pub_goal_point(self,_timer):
        """发布目标点位置作为PointStamped消息"""
        goal_msg = PointStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = 'world'  # 使用与其他可视化相同的坐标系
        goal_msg.point.x = self.noisy_goal[0]
        goal_msg.point.y = self.noisy_goal[1]
        goal_msg.point.z = self.noisy_goal[2]
        self.goal_point_pub.publish(goal_msg)
    
    def init_fov_visual(self):
        # 从配置文件中读取相机参数
        fx = self.cfg['camera']['fx']
        fy = self.cfg['camera']['fy']
        cx = self.cfg['camera']['cx']
        cy = self.cfg['camera']['cy']
        image_width = self.cfg['camera']['image_width']
        image_height = self.cfg['camera']['image_height']
        max_depth_dist = self.cfg['camera']['max_depth_dist'] / 4  # 不直接看到最远的地方(20m)，直接只可视化20/4=5m的范围

        # 计算 FoV 顶点（相机坐标系下）
        self.fov_node = [
            np.array([0, 0, 0]),  # 相机位置
            np.array([max_depth_dist,(image_width - cx) * max_depth_dist / fx, (image_height - cy) * max_depth_dist / fy]),  # 左上角
            np.array([max_depth_dist, (0 - cx) * max_depth_dist / fx, (image_height - cy) * max_depth_dist / fy]),  # 右上角
            np.array([max_depth_dist, (0 - cx) * max_depth_dist / fx, (0 - cy) * max_depth_dist / fy]),  # 右下角
            np.array([max_depth_dist, (image_width - cx) * max_depth_dist / fx,  (0 - cy) * max_depth_dist / fy]),  # 左下角
        ]

        # 初始化 Marker
        self.markerNode_fov = Marker()
        self.markerNode_fov.header.frame_id = "drone2"
        self.markerNode_fov.type = Marker.SPHERE_LIST
        self.markerNode_fov.ns = "fov_nodes"
        self.markerNode_fov.scale.x = 0.05
        self.markerNode_fov.scale.y = 0.05
        self.markerNode_fov.scale.z = 0.05
        self.markerNode_fov.color.r = 0
        self.markerNode_fov.color.g = 0.8
        self.markerNode_fov.color.b = 1
        self.markerNode_fov.color.a = 1

        self.markerEdge_fov = Marker()
        self.markerEdge_fov.header.frame_id = "drone2"
        self.markerEdge_fov.type = Marker.LINE_LIST
        self.markerEdge_fov.ns = "fov_edges"
        self.markerEdge_fov.scale.x = 0.05
        self.markerEdge_fov.color.r = 0.5
        self.markerEdge_fov.color.g = 0.0
        self.markerEdge_fov.color.b = 0.0
        self.markerEdge_fov.color.a = 1

        # ROS 发布器
        self.fov_visual_pub = rospy.Publisher("/sensor_fov_visual", MarkerArray, queue_size=1)
    
    def initialize_kalman_filter(self):
        """初始化卡尔曼滤波器参数，使用匀速运动模型"""
        # 状态向量: [x, y, z, vx, vy, vz]^T
        self.kalman_state = np.zeros(6)
        # 初始化位置为目标初始位置
        self.kalman_state[0:3] = self.goal
        
        # 状态协方差矩阵，反映我们对状态估计的不确定性
        self.kalman_P = np.eye(6) * 0.1  # 初始不确定性
        
        # 状态转移矩阵，基于匀速运动模型
        dt = self.ctrl_dt  # 控制周期
        self.kalman_F = np.eye(6)
        self.kalman_F[0, 3] = dt  # x = x + vx*dt
        self.kalman_F[1, 4] = dt  # y = y + vy*dt
        self.kalman_F[2, 5] = dt  # z = z + vz*dt
        
        # 过程噪声协方差矩阵，反映模型不确定性
        process_noise = 0.01  # 过程噪声大小
        self.kalman_Q = np.eye(6) * process_noise
        self.kalman_Q[3:6, 3:6] *= 2  # 速度的过程噪声稍大
        
        # 观测矩阵，我们只观测位置，不直接观测速度
        self.kalman_H = np.zeros((3, 6))
        self.kalman_H[0, 0] = 1
        self.kalman_H[1, 1] = 1
        self.kalman_H[2, 2] = 1
        
        # 观测噪声协方差矩阵
        measurement_noise = 0.05  # 观测噪声大小
        self.kalman_R = np.eye(3) * measurement_noise
        
        # 时间戳，用于计算实际的时间间隔
        self.last_kalman_time = rospy.Time.now()
    
    def kalman_predict(self, dt=None):
        """卡尔曼滤波器预测步骤"""
        # 使用实际的时间间隔，而不是固定的控制周期
        if dt is None:
            current_time = rospy.Time.now()
            dt = (current_time - self.last_kalman_time).to_sec()
            self.last_kalman_time = current_time
        
        # 更新状态转移矩阵中的时间间隔
        F = np.copy(self.kalman_F)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        
        # 预测状态
        self.kalman_state = F @ self.kalman_state
        
        # 预测协方差
        self.kalman_P = F @ self.kalman_P @ F.T + self.kalman_Q
        
        # 更新noisy_goal为预测的位置
        self.noisy_goal = self.kalman_state[0:3].copy()
        print("self.noisy_goal:",self.noisy_goal)

    def kalman_update(self, measurement):
        """卡尔曼滤波器更新步骤"""
        # 计算卡尔曼增益
        S = self.kalman_H @ self.kalman_P @ self.kalman_H.T + self.kalman_R
        K = self.kalman_P @ self.kalman_H.T @ np.linalg.inv(S)
        
        # 更新状态
        residual = measurement - self.kalman_H @ self.kalman_state
        self.kalman_state = self.kalman_state + K @ residual
        
        # 更新协方差
        I = np.eye(6)
        self.kalman_P = (I - K @ self.kalman_H) @ self.kalman_P
        
        # 更新noisy_goal为估计的位置
        self.noisy_goal = self.kalman_state[0:3].copy()
        
        # 更新目标可见状态
        self.is_target_visible = True
    
    # 边界探索算法
    def search_for_target(self):
        """基于边界探索的目标搜索算法"""
        current_time = rospy.Time.now()
        
        # 定期更新搜索目标
        # 检查是否到达当前搜索目标
        drone_pos = np.array([self.odom.pose.pose.position.x, 
                             self.odom.pose.pose.position.y, 
                             self.odom.pose.pose.position.z])
        
        # 计算到当前目标点的距离
        distance_to_goal = np.linalg.norm(drone_pos[:2] - self.noisy_goal[:2])
        
        # 如果距离小于阈值，认为已到达目标点，需要更新下一个目标
        if distance_to_goal < 5.0:  # 3米的到达阈值
            self.last_search_time = current_time
            
            # 查找未被探索的区域作为目标点
            unexplored_points = np.argwhere(self.voxel_map == 0)
            
            if len(unexplored_points) > 0:
                # 转换为世界坐标
                drone_pos = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y])
                
                # 计算每个未探索点到无人机当前位置的距离
                unexplored_world = []
                for point in unexplored_points:
                    map_y, map_x = point
                    world_x = self.map_origin[0] + map_x * self.map_resolution
                    world_y = self.map_origin[1] + map_y * self.map_resolution
                    unexplored_world.append((world_x, world_y))
                
                unexplored_world = np.array(unexplored_world)
                distances = np.linalg.norm(unexplored_world - drone_pos, axis=1)
                
                # 筛选距离在搜索半径内的点
                valid_indices = np.where(distances <= self.search_radius)[0]
                
                if len(valid_indices) > 0:
                    # 按距离排序，优先选择较远的点进行探索
                    sorted_indices = valid_indices[np.argsort(distances[valid_indices])[-min(10, len(valid_indices)):]]
                    
                    # 随机选择一个点作为目标
                    selected_idx = np.random.choice(sorted_indices)
                    target_x, target_y = unexplored_world[selected_idx]
                    
                    # 设置新的搜索目标
                    self.noisy_goal = np.array([target_x, target_y, self.pos[2]])
                    # print(f"设置新的搜索目标: ({target_x:.2f}, {target_y:.2f})")
                    return
            
            # 如果没有合适的未探索区域，使用螺旋探索策略
            self.exploration_direction = (self.exploration_direction + self.exploration_step) % 360
            angle_rad = np.radians(self.exploration_direction)
            
            # 计算螺旋搜索点
            search_x = self.pos[0] + self.search_radius * np.cos(angle_rad)
            search_y = self.pos[1] + self.search_radius * np.sin(angle_rad)
            
            # 设置新的螺旋搜索目标
            self.noisy_goal = np.array([search_x, search_y, self.pos[2]])
            print(f"使用螺旋搜索: 方向={self.exploration_direction}度")



    def pub_fov_visual(self):
        if not self.odom_init:
            return
        
        # 更新 Marker
        self.markerNode_fov.points = []
        self.markerEdge_fov.points = []
        for node in self.fov_node:
            point = Point()
            point.x, point.y, point.z = node
            self.markerNode_fov.points.append(point)

        # 添加边界线
        edges = [
            (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 2), (2, 3), (3, 4), (4, 1)
        ]
        for edge in edges:
            self.markerEdge_fov.points.append(self.markerNode_fov.points[edge[0]])
            self.markerEdge_fov.points.append(self.markerNode_fov.points[edge[1]])

        # 发布 MarkerArray
        marker_array = MarkerArray()
        marker_array.markers.append(self.markerNode_fov)
        marker_array.markers.append(self.markerEdge_fov)
        self.fov_visual_pub.publish(marker_array)


    def callback_set_goal(self, data):
        # self.goal = np.asarray([data.pose.pose.position.x, data.pose.pose.position.y, 1])
        self.goal = np.asarray([data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z])
        # 更新球体中心为新目标位置
        self.target_sphere_center = self.goal
        # 仅平移预生成的相对坐标点，避免重新生成球体
        self.target_sphere_points = self.target_sphere_relative_points + self.target_sphere_center
        
        
        # 修改：降低触发到达状态的距离阈值，从4.5米改为2米
        if np.linalg.norm(self.pos - self.noisy_goal) > 4.0:
            self.arrive = False
        # print(f"New Goal: ({data.pose.pose.position.x:.1f}, {data.pose.pose.position.y:.1f})")

    # the first frame
    def callback_odometry(self, data):
        self.odom = data
        if not self.desire_init:
            self.desire_pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            self.desire_vel = np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.desire_acc = np.array((0.0, 0.0, 0.0))
            ypr = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                               self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_euler('ZYX', degrees=False)
            self.last_yaw = ypr[0]
        self.odom_init = True

        self.pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        if np.linalg.norm(self.pos - self.noisy_goal) < 5 and not self.arrive:
            # print("Arrive!")
            self.arrive = True
        
        # 发布筛选后的odom数据
        current_pos = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z])
        
        if not self.filtered_odom_first_pub or np.linalg.norm(current_pos - self.last_published_pos) >= self.position_threshold:
            # 添加点到累积列表
            self.trajectory_points.append(current_pos)
            # if self.last_published_pos is not None:
            #     self.dis_count += np.linalg.norm(current_pos - self.last_published_pos)
                
            # 更新上次发布位置和标记
            self.last_published_pos = current_pos.copy()
            self.filtered_odom_first_pub = True
        
        # 创建点云消息
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'world'  # 使用与odom相同的坐标系
        # 转换点列表为点云格式
        points_array = np.array(self.trajectory_points, dtype=np.float32)
        # 创建PointCloud2消息
        point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
        # 发布点云消息
        self.filtered_traj_pub.publish(point_cloud_msg)
        # print(f"dis_count: {self.dis_count:.2f}")


    def process_odom(self):
        # Rwb -> Rwc -> Rcw
        Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                   self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
        self.Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)
        Rotation_cw = self.Rotation_wc.T

        # vel and acc
        vel_w = self.desire_vel if self.plan_from_reference else np.array([self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z])
        vel_c = np.dot(Rotation_cw, vel_w)
        acc_w = self.desire_acc
        acc_c = np.dot(Rotation_cw, acc_w)

        # goal_dir
        # 目标位置添加高斯白噪声
        # goal_noise_sigma = 0.2  # 噪声标准差，可根据需要调整
        # noisy_goal = self.goal + np.random.normal(0, goal_noise_sigma, size=self.goal.shape)
        # noisy_goal = self.goal
        # print("Noisy Goal:", noisy_goal)
        # print("self.goal:", self.goal)
        # if self.is_mask_vis:
        #     self.noisy_goal = noisy_goal

        goal_w = self.noisy_goal - self.desire_pos
        goal_c = np.dot(Rotation_cw, goal_w)

        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm.to(self.device, non_blocking=True)
    
    @torch.inference_mode()
    def callback_depth_pointcloud(self):
        if not self.odom_init:
            return

        # 1. 深度图处理
        with self.depth_share_lock:
            if self.depth_share is None:
                return
        
        depth = self.depth_share 

        # 2. 生成点云
        fu, fv = self.cfg['camera']['fx'], self.cfg['camera']['fy']
        cu, cv = self.cfg['camera']['cx'], self.cfg['camera']['cy']
        rows, cols = depth.shape
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))  # 图像坐标系（左上角为原点）
        v = rows - 1 - v  # 翻转 v，使其从图像的下方开始
        u = cols - 1 - u  # 翻转 u，使其从图像的下方开始
        x = depth
        y = (u - cu) * depth / fu
        z = (v - cv) * depth / fv
        points_camera = np.stack((x, y, z), axis=-1).reshape(-1, 3) 

        valid_depth = np.logical_and(points_camera[:, 0] > 0, points_camera[:, 0] < self.max_dis)
        points_camera = points_camera[valid_depth]

        # 4. 生成点云消息
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "drone2"
        point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_camera)

        # 5. 发布点云
        self.fov_pub.publish(point_cloud_msg)

        if self.search_mode:    # search 模式下才对地图进行更新
            # 4. 转换到世界坐标系
            drone_position = np.array([self.odom.pose.pose.position.x,
                                        self.odom.pose.pose.position.y,
                                        self.odom.pose.pose.position.z])
            points_world = np.dot(self.Rotation_wc, points_camera.T).T + drone_position   # 世界坐标系中的目标位置
            # # 过滤高度在0.2-0.5米之间的点
            # height_mask = np.logical_and(points_world[:, 2] >= 1, points_world[:, 2] <= 1.5)
            # points_world = points_world[height_mask]

            # 6. 更新2D voxel map
            if len(points_world) > 0:
                # 将世界坐标转换为地图坐标
                map_x = ((points_world[:, 0] - self.map_origin[0]) / self.map_resolution).astype(int)
                map_y = ((points_world[:, 1] - self.map_origin[1]) / self.map_resolution).astype(int)
                
                # 过滤地图范围内的点
                valid_map_coords = np.logical_and.reduce((map_x >= 0, map_x < self.map_size, 
                                                        map_y >= 0, map_y < self.map_size))
                map_x = map_x[valid_map_coords]
                map_y = map_y[valid_map_coords]
                
                # 更新地图（将对应格子标记为已经探索过）
                self.voxel_map[map_y, map_x] = 1  # 标记为障碍物

    # 添加发布2D地图的方法
    def publish_2d_map(self, _timer):
        """将2D voxel map转换为图像并发布"""
        # 将地图转换为图像（0=黑色，1=白色）
        map_image = np.uint8(self.voxel_map * 255)  # 0=未知(黑色), 255=障碍物(白色)
        
        # 创建ROS图像消息
        map_msg = self.bridge.cv2_to_imgmsg(map_image, encoding="mono8")
        map_msg.header.frame_id = "world"
        map_msg.header.stamp = rospy.Time.now()
        
        # 发布地图
        self.map_pub.publish(map_msg)

    @torch.inference_mode()
    def callback_depth(self, data):
        if not self.odom_init: return

        # 1. Depth Image Process
        try:
            depth = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except:
            print("CV_bridge ERROR")

        time0 = time.time()
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = np.minimum(depth * self.scale, self.max_dis)
        depth_mask =  np.zeros_like(depth, dtype=np.float32)

        # 将目标球体点云投影到深度图
        if self.target_sphere_points is not None and self.odom_init:
            # 获取相机参数
            fx = self.cfg['camera']['fx']
            fy = self.cfg['camera']['fy']
            cx = self.cfg['camera']['cx']
            cy = self.cfg['camera']['cy']
            
            # 计算旋转矩阵(世界坐标系到相机坐标系)
            Rotation_wb = R.from_quat([self.odom.pose.pose.orientation.x, self.odom.pose.pose.orientation.y,
                                      self.odom.pose.pose.orientation.z, self.odom.pose.pose.orientation.w]).as_matrix()
            Rotation_wc = np.dot(Rotation_wb, self.Rotation_bc)  # 世界到相机的旋转矩阵
            Rotation_cw = Rotation_wc.T  # 相机到世界的旋转矩阵
            
            # 无人机位置
            drone_position = np.array([self.odom.pose.pose.position.x, 
                                      self.odom.pose.pose.position.y, 
                                      self.odom.pose.pose.position.z])
            
            # 将球体点云从世界坐标系转换到相机坐标系
            sphere_points_world = self.target_sphere_points
            sphere_points_camera = np.dot(Rotation_cw, (sphere_points_world - drone_position).T).T
            
            # 过滤相机前方的点(x > 0)
            valid_indices = sphere_points_camera[:, 0] > 0
            sphere_points_camera = sphere_points_camera[valid_indices]
            
            # 将3D点投影到2D图像平面
            u = (sphere_points_camera[:, 1] * fx / sphere_points_camera[:, 0]) + cx
            v = (sphere_points_camera[:, 2] * fy / sphere_points_camera[:, 0]) + cy
            u = (self.width - 1 - u)   # 翻转u坐标（图像宽度方向）
            v = self.height - 1 - v  # 翻转v坐标（图像高度方向）
            
            # 创建有效的坐标掩码（仅保留图像范围内的点）
            valid_mask = (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
            # 应用掩码过滤所有相关数组
            u = u[valid_mask].astype(int)
            v = v[valid_mask].astype(int)
            sphere_points_camera = sphere_points_camera[valid_mask]
            # 初始化可见性掩码 (标记哪些球体像素实际可见)
            visibility_mask = np.zeros_like(u, dtype=bool)
            
            # 更新深度图，添加球体的深度信息（向量化优化）
            indices = v * self.width + u
            depth_flat = depth.ravel()
            # 创建更新掩码：只更新NaN区域或比现有深度更近的点
            mask = np.isnan(depth_flat[indices]) | (depth_flat[indices] > sphere_points_camera[:, 0])
            # 应用深度更新
            depth_flat[indices[mask]] = sphere_points_camera[mask, 0]
            # 将展平数组重塑并写回原始深度图
            depth = depth_flat.reshape(depth.shape)
            # 更新可见性掩码
            visibility_mask[mask] = True

            # 仅绘制可见的球体像素（应用可见性掩码过滤遮挡像素）
            depth_mask[v[visibility_mask], u[visibility_mask]] = 1 # 表示该区域有球体投影
            
            # 保存球体投影坐标用于RGB绘制 (使用线程锁确保安全)
            with self.sphere_uv_lock:
                self.sphere_uv = (u, v, visibility_mask)

        with self.depth_share_lock:
            self.depth_share = depth * 1.0
            self.depth_mask = depth_mask * 1.0

        depth = depth / self.max_dis

        # interpolated the nan value (experiment shows that treating nan directly as 0 produces similar results)
        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        interpolated_image = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
        interpolated_image = interpolated_image.astype(np.float32) / 255.0
        depth = interpolated_image.reshape([1, 1, self.height, self.width])
        depth_mask = cv2.resize(depth_mask, (self.mask_height, self.mask_width), interpolation=cv2.INTER_NEAREST)
        depth_mask = depth_mask.reshape([1, 1, self.mask_height, self.mask_width])
        # print("depth_ min:", np.min(depth), "max:", np.max(depth))
        # cv2.imshow("1", depth[0][0])
        # cv2.waitKey(1)

        # 2. UAST Network Inference
        # input prepare
        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)  # (non_blocking: copying speed 3x)
        depth_mask_input = torch.from_numpy(depth_mask).to(self.device, non_blocking=True)  # (non_blocking: copying speed 3x)
        obs_norm = self.process_odom()
        obs_input = self.state_transform.prepare_input(obs_norm)
        obs_input = obs_input.to(self.device, non_blocking=True)
        # torch.cuda.synchronize()

        time2 = time.time()
        # Forward (TensorRT: inference speed increased by 5x)
        endstate_pred, score_pred = self.policy(depth_input, depth_mask_input, obs_input)
        endstate_pred, score_pred = endstate_pred.cpu().numpy(), score_pred.cpu().numpy()
        time3 = time.time()
        # Replacing PyTorch operation on CUDA with NumPy operation on CPU (speed increased by 10x)
        endstate, score = self.process_output(endstate_pred, score_pred, return_all_preds=self.visualize)
        # Vectorization: transform the prediction(P V A in body frame) to the world frame with the attitude (without the position)
        endstate_c = endstate.reshape(-1, 3, 3).transpose(0, 2, 1)  # [N, 9] -> [N, 3, 3] -> [px vx ax, py vy ay, pz vz az]
        endstate_w = np.matmul(self.Rotation_wc, endstate_c)

        action_id = np.argmin(score_pred) if self.visualize else 0
        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety
            start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
            start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
            self.optimal_poly_x = Poly5Solver(start_pos[0], start_vel[0], self.desire_acc[0], endstate_w[action_id, 0, 0] + start_pos[0],
                                              endstate_w[action_id, 0, 1], endstate_w[action_id, 0, 2], self.traj_time)
            self.optimal_poly_y = Poly5Solver(start_pos[1], start_vel[1], self.desire_acc[1], endstate_w[action_id, 1, 0] + start_pos[1],
                                              endstate_w[action_id, 1, 1], endstate_w[action_id, 1, 2], self.traj_time)
            self.optimal_poly_z = Poly5Solver(start_pos[2], start_vel[2], self.desire_acc[2], endstate_w[action_id, 2, 0] + start_pos[2],
                                              endstate_w[action_id, 2, 1], endstate_w[action_id, 2, 2], self.traj_time)
            self.ctrl_time = 0.0
        time4 = time.time()
        self.visualize_trajectory(score_pred, endstate_w)
        time5 = time.time()
        self.pub_fov_visual()   # publish the sensor FOV visualization
        self.callback_depth_pointcloud()

        if self.verbose:
            self.time_interpolation = self.time_interpolation + (time1 - time0)
            self.time_prepare = self.time_prepare + (time2 - time1)
            self.time_forward = self.time_forward + (time3 - time2)
            self.time_process = self.time_process + (time4 - time3)
            self.time_visualize = self.time_visualize + (time5 - time4)
            self.count = self.count + 1
            print(f"Time Consuming:"
                  f"depth-interpolation: {1000 * self.time_interpolation / self.count:.2f}ms;"
                  f"data-prepare: {1000 * self.time_prepare / self.count:.2f}ms; "
                  f"network-inference: {1000 * self.time_forward / self.count:.2f}ms; "
                  f"post-process: {1000 * self.time_process / self.count:.2f}ms;"
                  f"visualize-trajectory: {1000 * self.time_visualize / self.count:.2f}ms")

    def callback_rgb(self, data):
        """RGB图像回调函数,复用球体投影坐标绘制红色球体"""
        if not self.odom_init:
            return
            
        # 获取保存的球体投影坐标
        with self.sphere_uv_lock:
            if self.sphere_uv is None:
                return
            u, v, visibility_mask = self.sphere_uv
        
        with self.depth_share_lock:
            if self.depth_share is None:
                return
        
        # 转换RGB图像
        try:
            rgb_image = self.bridge.imgmsg_to_cv2(data, "8UC3")
        except Exception as e:
            rospy.logerr(f"RGB图像转换错误: {e}")
            return
            
        # 调整RGB图像尺寸以匹配投影坐标
        if rgb_image.shape[0] != self.height or rgb_image.shape[1] != self.width:
            rgb_image = cv2.resize(rgb_image, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # 仅绘制可见的球体像素（应用可见性掩码过滤遮挡像素
        visible_u = u[visibility_mask]
        visible_v = v[visibility_mask]
        rgb_image[visible_v, visible_u] = [0, 0, 255]  # BGR格式红色

        # 卡尔曼滤波预测目标位置
        # self.is_target_visible = False
        # self.kalman_predict() 

        # 从深度图中获取目标的深度值
        target_depth = self.depth_share[visible_v, visible_u]  # 恢复原始深度值
        target_depth_valid_mask = (target_depth > 0)
        if np.sum(target_depth_valid_mask) > 8:
            target_u = visible_u[target_depth_valid_mask]
            target_v = visible_v[target_depth_valid_mask]
            target_u = self.width - 1 - target_u   # 翻转u坐标（图像宽度方向）
            target_v = self.height - 1 - target_v  # 翻转v坐标（图像高度方向）
            target_depth = target_depth[target_depth_valid_mask]

            # 获得相机坐标系下的可见目标的3D点
            fx, fy = self.cfg['camera']['fx'], self.cfg['camera']['fy']
            cx, cy = self.cfg['camera']['cx'], self.cfg['camera']['cy']
            target_x = target_depth
            target_y = (target_u - cx) * target_depth / fx
            target_z = (target_v - cy) * target_depth / fy

            target_camera = np.array([target_x.mean(), target_y.mean(), target_z.mean()])  # 相机坐标系中的目标位置

            # 4. 将目标从相机坐标系转换到世界坐标系
            drone_position = np.array([self.odom.pose.pose.position.x,
                                        self.odom.pose.pose.position.y,
                                        self.odom.pose.pose.position.z])
            target_world = (np.dot(self.Rotation_wc, target_camera.T) + drone_position.T).T  # 世界坐标系中的目标位置
            self.noisy_goal = target_world
            # 目标位置添加高斯白噪声
            # goal_noise_sigma = 0.2  # 噪声标准差，可根据需要调整
            # self.noisy_goal = target_world + np.random.normal(0, goal_noise_sigma, size=self.goal.shape)
            self.last_seen_time = rospy.Time.now()
            #  # 使用卡尔曼滤波器更新状态
            # self.kalman_update(target_world)  # 更新
            if self.search_mode:
                print("目标重新可见，退出搜索模式.")
                self.search_mode = False  # 目标可见，退出搜索模式
        else:
            # 检查是否需要进入搜索模式
            current_time = rospy.Time.now()
            if (current_time - self.last_seen_time).to_sec() > 2.0 and not self.search_mode:
                self.search_mode = True
                self.voxel_map *= 0  # 重置地图
                print("进入搜索模式...")                                
        
        # 发布带球体的RGB图像
        rgb_msg = self.bridge.cv2_to_imgmsg(rgb_image, "bgr8")
        rgb_msg.header.stamp = data.header.stamp
        rgb_msg.header.frame_id = "drone2"
        self.rgb_vis_target_pub.publish(rgb_msg)

        # 发布带球体的深度图像
        depth_msg = self.bridge.cv2_to_imgmsg(self.depth_share, encoding="32FC1")
        depth_msg.header.stamp = data.header.stamp
        depth_msg.header.frame_id = "drone2"  # 使用与相机相同的坐标系
        self.depth_vis_target_pub.publish(depth_msg)

        # 发布带球体的掩膜图像
        mask_msg = self.bridge.cv2_to_imgmsg(self.depth_mask, encoding="32FC1")
        mask_msg.header.stamp = data.header.stamp
        mask_msg.header.frame_id = "drone2"  # 使用与相机相同的坐标系
        self.mask_vis_target_pub.publish(mask_msg)

    def control_pub(self, _timer):
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            return

        # 在搜索模式下，更新搜索目标
        if self.search_mode:
            self.search_for_target()
            # self.search_count += 1
        # else:
        #     # self.track_count += 1
        
        # print(f"search_count: {self.search_count}, track_count: {self.track_count}, all_count: {self.search_count + self.track_count}")

        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False   # ready for next rollout
            # self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            # self.ctrl_pub.publish(self.last_control_msg)
            
            # 目标在范围内时，始终朝向目标悬停
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_EMPTY
            control_msg.position.x = self.odom.pose.pose.position.x
            control_msg.position.y = self.odom.pose.pose.position.y
            control_msg.position.z = self.odom.pose.pose.position.z
            desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            goal_dir = self.noisy_goal - self.desire_pos
            goal_yaw = np.arctan2(goal_dir[1], goal_dir[0])
            control_msg.yaw = goal_yaw
            self.ctrl_pub.publish(control_msg)

            return

        with self.lock:  # Python3.8: threads are scheduled using time slices, add the lock to ensure safety and publish frequency
            self.ctrl_time += self.ctrl_dt
            control_msg = PositionCommand()
            control_msg.header.stamp = rospy.Time.now()
            control_msg.trajectory_flag = control_msg.TRAJECTORY_STATUS_READY
            control_msg.position.x = self.optimal_poly_x.get_position(self.ctrl_time)
            control_msg.position.y = self.optimal_poly_y.get_position(self.ctrl_time)
            control_msg.position.z = self.optimal_poly_z.get_position(self.ctrl_time)
            control_msg.velocity.x = self.optimal_poly_x.get_velocity(self.ctrl_time)
            control_msg.velocity.y = self.optimal_poly_y.get_velocity(self.ctrl_time)
            control_msg.velocity.z = self.optimal_poly_z.get_velocity(self.ctrl_time)
            control_msg.acceleration.x = self.optimal_poly_x.get_acceleration(self.ctrl_time)
            control_msg.acceleration.y = self.optimal_poly_y.get_acceleration(self.ctrl_time)
            control_msg.acceleration.z = self.optimal_poly_z.get_acceleration(self.ctrl_time)
            self.desire_pos = np.array([control_msg.position.x, control_msg.position.y, control_msg.position.z])
            self.desire_vel = np.array([control_msg.velocity.x, control_msg.velocity.y, control_msg.velocity.z])
            self.desire_acc = np.array([control_msg.acceleration.x, control_msg.acceleration.y, control_msg.acceleration.z])
            goal_dir = self.noisy_goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw_track(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
            self.last_yaw = yaw
            control_msg.yaw = yaw
            control_msg.yaw_dot = yaw_dot
            self.desire_init = True
            self.last_control_msg = control_msg
            self.ctrl_pub.publish(control_msg)
            # print("control_msg:",control_msg)

    def process_output(self, endstate_pred, score_pred, return_all_preds=False):
        endstate_pred = endstate_pred.reshape(9, self.lattice_primitive.traj_num).T
        score_pred = score_pred.reshape(self.lattice_primitive.traj_num)

        if not return_all_preds:
            action_id = np.argmin(score_pred)
            lattice_id = self.lattice_primitive.traj_num - 1 - action_id
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred[action_id, :][np.newaxis, :], lattice_id)
            score = score_pred[action_id]
        else:
            score = score_pred
            endstate = self.state_transform.pred_to_endstate_cpu(endstate_pred, torch.arange(self.lattice_primitive.traj_num-1, -1, -1))

        return endstate, score

    def visualize_trajectory(self, pred_score, pred_endstate):
        dt = self.traj_time / 20.0
        start_pos = self.desire_pos if self.plan_from_reference else np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        start_vel = self.desire_vel if self.plan_from_reference else np.array((self.odom.twist.twist.linear.x, self.odom.twist.twist.linear.y, self.odom.twist.twist.linear.z))
        # best predicted trajectory
        if self.best_traj_pub.get_num_connections() > 0:
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                self.optimal_poly_x.get_position(t_values),
                self.optimal_poly_y.get_position(t_values),
                self.optimal_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.best_traj_pub.publish(point_cloud_msg)
        # lattice primitive
        if self.visualize and self.lattice_traj_pub.get_num_connections() > 0:
            lattice_endstate = self.lattice_primitive.lattice_pos_node.cpu().numpy()
            lattice_endstate = np.dot(lattice_endstate, self.Rotation_wc.T)
            zero_state = np.zeros_like(lattice_endstate)
            lattice_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                          lattice_endstate[:, 0] + start_pos[0], zero_state[:, 0], zero_state[:, 0], self.traj_time)
            lattice_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                          lattice_endstate[:, 1] + start_pos[1], zero_state[:, 1], zero_state[:, 1], self.traj_time)
            lattice_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                          lattice_endstate[:, 2] + start_pos[2], zero_state[:, 2], zero_state[:, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                lattice_poly_x.get_position(t_values),
                lattice_poly_y.get_position(t_values),
                lattice_poly_z.get_position(t_values)
            ), axis=-1)
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            point_cloud_msg = point_cloud2.create_cloud_xyz32(header, points_array)
            self.lattice_traj_pub.publish(point_cloud_msg)
        # all predicted trajectories
        if self.visualize and self.all_trajs_pub.get_num_connections() > 0:
            all_poly_x = Polys5Solver(start_pos[0], start_vel[0], self.desire_acc[0],
                                      pred_endstate[:, 0, 0] + start_pos[0], pred_endstate[:, 0, 1], pred_endstate[:, 0, 2], self.traj_time)
            all_poly_y = Polys5Solver(start_pos[1], start_vel[1], self.desire_acc[1],
                                      pred_endstate[:, 1, 0] + start_pos[1], pred_endstate[:, 1, 1], pred_endstate[:, 1, 2], self.traj_time)
            all_poly_z = Polys5Solver(start_pos[2], start_vel[2], self.desire_acc[2],
                                      pred_endstate[:, 2, 0] + start_pos[2], pred_endstate[:, 2, 1], pred_endstate[:, 2, 2], self.traj_time)
            t_values = np.arange(0, self.traj_time, dt)
            points_array = np.stack((
                all_poly_x.get_position(t_values),
                all_poly_y.get_position(t_values),
                all_poly_z.get_position(t_values)
            ), axis=-1)
            scores = np.repeat(pred_score, t_values.size)
            points_array = np.column_stack((points_array, scores))
            header = std_msgs.msg.Header()
            header.stamp = rospy.Time.now()
            header.frame_id = 'world'
            fields = [PointField('x', 0, PointField.FLOAT32, 1), PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), PointField('intensity', 12, PointField.FLOAT32, 1)]
            point_cloud_msg = point_cloud2.create_cloud(header, fields, points_array)
            self.all_trajs_pub.publish(point_cloud_msg)

    def warm_up(self):
        depth = torch.zeros((1, 1, self.height, self.width), dtype=torch.float32, device=self.device)
        depth_mask = torch.zeros((1, 1, self.mask_height, self.mask_width), dtype=torch.float32, device=self.device)
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred = self.policy(depth, depth_mask, obs)
        _ = self.state_transform.pred_to_endstate(endstate_pred)


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_tensorrt", type=int, default=0, help="use tensorrt or not")
    parser.add_argument("--trial", type=int, default=1, help="trial number")
    parser.add_argument("--epoch", type=int, default=50, help="epoch number")
    return parser


def main():
    args = parser().parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    weight = base_dir + "/saved/UAST_track/epoch50.pth"
    print("load weight from:", weight)

    settings = {'use_tensorrt': args.use_tensorrt,
                'goal': [-45, -45, 2],      # 目标点位置
                'env': 'simulation',     # 深度图来源 ('435' or 'simulation', 和深度单位有关)
                'pitch_angle_deg': -0,   # 相机俯仰角(仰为负)
                'target_topic': '/drone1/sim/odom',                   # 里程计话题
                'odom_topic': '/drone2/sim/odom',                   # 里程计话题
                'depth_topic': '/drone2/depth_image',               # 深度图话题
                'rgb_topic': '/drone2/rgb_image',                   #  rgb图话题
                'ctrl_topic': '/drone2/so3_control/pos_cmd',        # 控制器话题
                'plan_from_reference': False,   # 从参考状态规划？位置控制器: True, 神经网络直接控制: False
                'verbose': False,               # 打印耗时？
                'visualize': True               # 可视化所有轨迹？(实飞改为False节省计算)
                }
    UASTNet(settings, weight)


if __name__ == "__main__":
    main()
