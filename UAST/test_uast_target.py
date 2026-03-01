import rospy
import std_msgs.msg
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, PointStamped
from cv_bridge import CvBridge
from threading import Lock
from sensor_msgs.msg import PointCloud2, PointField, Image
from sensor_msgs import point_cloud2

import cv2
import os
import time
import torch
import numpy as np
import argparse
from ruamel.yaml import YAML
from scipy.spatial.transform import Rotation as R

from control_msg import PositionCommand
from policy.uast_network import YopoNetwork
from policy.poly_solver import *
from policy.state_transform import *

try:
    from torch2trt import TRTModule
except ImportError:
    print("tensorrt not found.")


class YopoNet:
    def __init__(self, config, weight):
        self.config = config
        rospy.init_node('yopo_net', anonymous=False)
        # load params
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cfg = YAML().load(open(os.path.join(base_dir, "config/traj_opt_target.yaml"), 'r'))
        self.height = cfg['image_height']
        self.width = cfg['image_width']
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
        self.state_transform = StateTransform(cfg)
        self.lattice_primitive = LatticePrimitive.get_instance(cfg)
        self.traj_time = self.lattice_primitive.segment_time

        # eval
        self.time_forward = 0.0
        self.time_process = 0.0
        self.time_prepare = 0.0
        self.time_interpolation = 0.0
        self.time_visualize = 0.0
        self.count = 0

        # Load Network
        if self.use_trt:
            self.policy = TRTModule()
            self.policy.load_state_dict(torch.load(weight))
        else:
            state_dict = torch.load(weight, weights_only=True)
            self.policy = YopoNetwork()
            self.policy.load_state_dict(state_dict)
            self.policy = self.policy.to(self.device)
            self.policy.eval()
        self.warm_up()

        # ros publisher
        self.lattice_traj_pub = rospy.Publisher("/yopo_net/lattice_trajs_visual", PointCloud2, queue_size=1)
        self.best_traj_pub = rospy.Publisher("/yopo_net/best_traj_visual", PointCloud2, queue_size=1)
        self.all_trajs_pub = rospy.Publisher("/yopo_net/trajs_visual", PointCloud2, queue_size=1)
        self.ctrl_pub = rospy.Publisher(self.config["ctrl_topic"], PositionCommand, queue_size=1)
        # ros subscriber
        self.odom_sub = rospy.Subscriber(self.config['odom_topic'], Odometry, self.callback_odometry, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.config['depth_topic'], Image, self.callback_depth, queue_size=1)
        self.goal_sub = rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_set_goal, queue_size=1)

        # 添加用于显示筛选后odom轨迹的publisher
        self.filtered_traj_pub = rospy.Publisher("/drone1/filtered_traj", PointCloud2, queue_size=1)
        self.trajectory_points = []  # 存储轨迹点的列表
        # 初始化上次发布的位置和标记
        self.last_published_pos = None
        self.filtered_odom_first_pub = False
        self.position_threshold = 0.5  # 距离阈值，单位：米

        # 自动导航参数
        self.goal_radius = 4.0  # 到达目标点的半径范围（米）
        self.min_goal_distance = 20.0  # 新目标点与当前位置的最小距离（米）
        self.goal_bounds = {'x': [-48, 48], 'y': [-48, 48]}  # 目标点范围
        self.auto_nav_active = False  # 自动导航标志

        np.random.seed(6)

        # 添加目标点发布器，用于可视化目标点
        self.goal_point_pub = rospy.Publisher("/yopo_net/goal_point", PointStamped, queue_size=1)
        
        # 添加drone2的odom订阅器
        self.drone2_odom_sub = rospy.Subscriber("/drone2/sim/odom", Odometry, self.callback_drone2_odometry, queue_size=1)

        # ros timer
        rospy.sleep(1.0)  # wait connection...
        self.timer_ctrl = rospy.Timer(rospy.Duration(self.ctrl_dt), self.control_pub)
        # 添加定时器用于检查目标点到达和发布目标点可视化
        self.timer_goal_check = rospy.Timer(rospy.Duration(0.1), self.check_goal_reached_and_publish)
        print("YOPO Net Node Ready!")
        rospy.spin()

    # 添加drone2的odom回调函数
    def callback_drone2_odometry(self, data):
        """处理drone2的odometry数据"""
        if self.auto_nav_active or not self.odom_init:
            return # 如果自动导航激活，则不处理drone2数据
        self.drone2_odom = data

        # 获取自身位置
        self_pos = np.array([self.odom.pose.pose.position.x, 
                            self.odom.pose.pose.position.y, 
                            self.odom.pose.pose.position.z])
        
        # 获取drone2位置
        drone2_pos = np.array([self.drone2_odom.pose.pose.position.x, 
                                self.drone2_odom.pose.pose.position.y, 
                                self.drone2_odom.pose.pose.position.z])
        # 计算距离
        if  np.linalg.norm(self_pos - drone2_pos) < 5:
            self.auto_nav_active = True # 如果drone2距离过近，则激活自动导航



        
    
    def generate_random_goal(self):
        """生成随机目标点，距离当前位置足够远，位于指定范围内"""
        if not self.odom_init:
            return None
        
        current_pos = np.array([self.odom.pose.pose.position.x, 
                               self.odom.pose.pose.position.y])
        
        # 尝试生成新目标点，直到找到距离足够远的点
        max_attempts = 50
        for _ in range(max_attempts):
            # 随机生成x,y坐标
            x = np.random.uniform(self.goal_bounds['x'][0], self.goal_bounds['x'][1])
            y = np.random.uniform(self.goal_bounds['y'][0], self.goal_bounds['y'][1])
            
            # 保持z坐标不变（当前高度）
            z = 1.5
            
            # 计算到当前位置的距离
            distance = np.linalg.norm(np.array([x, y]) - current_pos)
            
            # 检查是否距离足够远且在指定范围内
            if distance >= self.min_goal_distance:
                return np.array([x, y, z])
        
        # 如果尝试次数过多仍未找到合适点，则返回一个默认点
        print("Warning: Failed to find distant goal, using default")
        return np.array([current_pos[0] + 10, current_pos[1] + 10, 1.5])

    def check_goal_reached_and_publish(self, _timer):
        """检查是否到达目标点并发布目标点可视化"""
        # 发布目标点可视化
        if self.goal is not None:
            goal_msg = PointStamped()
            goal_msg.header.stamp = rospy.Time.now()
            goal_msg.header.frame_id = 'world'
            goal_msg.point.x = self.goal[0]
            goal_msg.point.y = self.goal[1]
            goal_msg.point.z = self.goal[2]
            self.goal_point_pub.publish(goal_msg)
        
        # 如果自动导航激活且还没有目标点，生成第一个目标点
        if self.auto_nav_active and self.goal is None and self.odom_init:
            self.goal = self.generate_random_goal()
            print(f"First random goal set: ({self.goal[0]:.1f}, {self.goal[1]:.1f})")
        
        # 检查是否到达目标点
        if self.odom_init and self.goal is not None:
            current_pos = np.array([self.odom.pose.pose.position.x, 
                                   self.odom.pose.pose.position.y, 
                                   self.odom.pose.pose.position.z])
            
            distance_to_goal = np.linalg.norm(current_pos - self.goal)
            
            # 如果在目标半径范围内，设置到达标志
            if distance_to_goal < self.goal_radius and not self.arrive:
                print(f"Arrived at goal ({self.goal[0]:.1f}, {self.goal[1]:.1f})")
                self.arrive = True
            
            # 如果已到达且自动导航激活，生成新目标
            if self.arrive and self.auto_nav_active:
                self.goal = self.generate_random_goal()
                self.arrive = False
                print(f"New random goal set: ({self.goal[0]:.1f}, {self.goal[1]:.1f})")
    

    def callback_set_goal(self, data):
        self.goal = np.asarray([data.pose.position.x, data.pose.position.y, 1.5])
        self.arrive = False
        print(f"New Goal: ({data.pose.position.x:.1f}, {data.pose.position.y:.1f})")

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

        pos = np.array((self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z))
        # if np.linalg.norm(pos - self.goal) < 5 and not self.arrive:
        #     print("Arrive!")
        #     self.arrive = True
        
        # 发布筛选后的odom数据
        current_pos = np.array([self.odom.pose.pose.position.x, self.odom.pose.pose.position.y, self.odom.pose.pose.position.z])
        if not self.filtered_odom_first_pub or np.linalg.norm(current_pos - self.last_published_pos) >= self.position_threshold:
            # 添加点到累积列表
            self.trajectory_points.append(current_pos)
            
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
        goal_w = self.goal - self.desire_pos
        goal_c = np.dot(Rotation_cw, goal_w)

        obs = np.concatenate((vel_c, acc_c, goal_c), axis=0).astype(np.float32)
        obs_norm = self.state_transform.normalize_obs(torch.from_numpy(obs[None, :]))
        return obs_norm.to(self.device, non_blocking=True)

    @torch.inference_mode()
    def callback_depth(self, data):
        if not self.odom_init: return

        # 1. Depth Image Process
        try:
            depth = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except:
            print("CV_bridge ERROR: Possible solutions may be found at https://github.com/TJU-Aerial-Robotics/YOPO/issues/2")

        time0 = time.time()
        if depth.shape[0] != self.height or depth.shape[1] != self.width:
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        depth = np.minimum(depth * self.scale, self.max_dis) / self.max_dis

        # interpolated the nan value (experiment shows that treating nan directly as 0 produces similar results)
        nan_mask = np.isnan(depth) | (depth < self.min_dis / self.max_dis)
        interpolated_image = cv2.inpaint(np.uint8(depth * 255), np.uint8(nan_mask), 1, cv2.INPAINT_NS)
        interpolated_image = interpolated_image.astype(np.float32) / 255.0
        depth = interpolated_image.reshape([1, 1, self.height, self.width])
        # print("depth_ min:", np.min(depth), "max:", np.max(depth))
        # cv2.imshow("1", depth[0][0])
        # cv2.waitKey(1)

        # 2. YOPO Network Inference
        # input prepare
        time1 = time.time()
        depth_input = torch.from_numpy(depth).to(self.device, non_blocking=True)  # (non_blocking: copying speed 3x)
        obs_norm = self.process_odom()
        obs_input = self.state_transform.prepare_input(obs_norm)
        obs_input = obs_input.to(self.device, non_blocking=True)
        # torch.cuda.synchronize()

        time2 = time.time()
        # Forward (TensorRT: inference speed increased by 5x)
        endstate_pred, score_pred = self.policy(depth_input, obs_input)
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

    def control_pub(self, _timer):
        if self.ctrl_time is None or self.ctrl_time > self.traj_time:
            return
        if self.arrive and self.last_control_msg is not None:
            self.desire_init = False   # ready for next rollout
            self.last_control_msg.trajectory_flag = self.last_control_msg.TRAJECTORY_STATUS_EMPTY
            self.ctrl_pub.publish(self.last_control_msg)
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
            goal_dir = self.goal - self.desire_pos
            yaw, yaw_dot = calculate_yaw(self.desire_vel, goal_dir, self.last_yaw, self.ctrl_dt)
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
        obs = torch.zeros((1, 9), dtype=torch.float32, device=self.device)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred = self.policy(depth, obs)
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
    weight = base_dir + "/saved/YOPO_nav/epoch50.pth"
    print("load weight from:", weight)

    settings = {'use_tensorrt': args.use_tensorrt,
                'goal': [45, 45, 2],      # 目标点位置
                'env': 'simulation',     # 深度图来源 ('435' or 'simulation', 和深度单位有关)
                'pitch_angle_deg': -0,   # 相机俯仰角(仰为负)
                'odom_topic': '/drone1/sim/odom',                   # 里程计话题
                'depth_topic': '/drone1/depth_image',               # 深度图话题
                'ctrl_topic': '/drone1/so3_control/pos_cmd',        # 控制器话题
                'plan_from_reference': False,   # 从参考状态规划？位置控制器: True, 神经网络直接控制: False
                'verbose': False,               # 打印耗时？
                'visualize': True               # 可视化所有轨迹？(实飞改为False节省计算)
                }
    YopoNet(settings, weight)


if __name__ == "__main__":
    main()
