import os
import torch
import numpy as np
from ruamel.yaml import YAML
from policy.primitive import LatticePrimitive


class StateTransform:
    def __init__(self,cfg=None):
        if cfg is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cfg = YAML().load(open(os.path.join(base_dir, "../config/traj_opt.yaml"), 'r'))
        self.lattice_primitive = LatticePrimitive.get_instance(cfg)
        self.goal_length = 2.0 * cfg['radio_range']

    def pred_to_endstate(self, endstate_pred: torch.Tensor) -> torch.Tensor:
        """
            Transform the predicted state to the body frame (Original prediction → Primitive frame → Body frame).
            endstate_pred: [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h]
            :return [batch; px py pz vx vy vz ax ay az; primitive_v; primitive_h] in body frame
        """
        B, N = endstate_pred.shape[0], endstate_pred.shape[2] * endstate_pred.shape[3]

        # [B, 9, 3, 5] -> [B, 3, 5, 9] -> [B, 15, 9]
        endstate_pred = endstate_pred.permute(0, 2, 3, 1).reshape(B, N, 9)

        # 获取 lattice angle 和 rotation (.flip: 由于lattice和grid的顺序相反)
        yaw, pitch = self.lattice_primitive.getAngleLattice()  # [15]
        yaw = yaw.flip(0)[None, :].expand(B, -1)  # [B, 15]
        pitch = pitch.flip(0)[None, :].expand(B, -1)  # [B, 15]
        Rbp = self.lattice_primitive.getRotation().flip(0)  # [15, 3, 3]
        Rbp = Rbp[None, :, :, :].expand(B, -1, -1, -1)  # [B, 15, 3, 3]

        delta_yaw = endstate_pred[:, :, 0] * self.lattice_primitive.yaw_diff  # [B, 15]
        delta_pitch = endstate_pred[:, :, 1] * self.lattice_primitive.pitch_diff
        radio = (endstate_pred[:, :, 2] + 1.0) * self.lattice_primitive.radio_range # 选择是小于radio_range还是大于的半径规划，[0,2*radio_range]

        cos_pitch = torch.cos(pitch + delta_pitch)
        endstate_x = cos_pitch * torch.cos(yaw + delta_yaw) * radio
        endstate_y = cos_pitch * torch.sin(yaw + delta_yaw) * radio
        endstate_z = torch.sin(pitch + delta_pitch) * radio
        endstate_p = torch.stack([endstate_x, endstate_y, endstate_z], dim=-1)  # [B, 15, 3]

        # vel / acc
        endstate_vp = endstate_pred[:, :, 3:6] * self.lattice_primitive.vel_max  # [B, 15, 3]
        endstate_ap = endstate_pred[:, :, 6:9] * self.lattice_primitive.acc_max  # [B, 15, 3]

        # v/a 变换到 body frame
        endstate_vb = torch.matmul(Rbp, endstate_vp.unsqueeze(-1)).squeeze(-1)  # [B, 15, 3]
        endstate_ab = torch.matmul(Rbp, endstate_ap.unsqueeze(-1)).squeeze(-1)

        endstate = torch.cat([endstate_p, endstate_vb, endstate_ab], dim=-1)  # [B, 15, 9]

        endstate = endstate.permute(0, 2, 1).reshape(B, 9, 3, 5)  # [B, 9, 3, 5]
        return endstate

    def pred_to_endstate_cpu(self, endstate_pred: np.ndarray, lattice_id: torch.Tensor) -> np.ndarray:
        """
            Used during test:
            Numpy version of pred_to_endstate() on CPU (used in test, x10 times faster than torch on CUDA)
            :return [B; px py pz vx vy vz ax ay az] in body frame
        """
        delta_yaw = endstate_pred[:, 0] * self.lattice_primitive.yaw_diff
        delta_pitch = endstate_pred[:, 1] * self.lattice_primitive.pitch_diff
        radio = (endstate_pred[:, 2] + 1.0) * self.lattice_primitive.radio_range

        yaw, pitch = self.lattice_primitive.getAngleLattice(lattice_id)
        yaw, pitch = yaw.cpu().numpy(), pitch.cpu().numpy()
        endstate_x = np.cos(pitch + delta_pitch) * np.cos(yaw + delta_yaw) * radio
        endstate_y = np.cos(pitch + delta_pitch) * np.sin(yaw + delta_yaw) * radio
        endstate_z = np.sin(pitch + delta_pitch) * radio
        endstate_p = np.stack((endstate_x, endstate_y, endstate_z), axis=1)

        endstate_vp = endstate_pred[:, 3:6] * self.lattice_primitive.vel_max
        endstate_ap = endstate_pred[:, 6:9] * self.lattice_primitive.acc_max

        Rpb = self.lattice_primitive.getRotation(lattice_id).cpu().numpy()
        endstate_vb = np.matmul(Rpb, endstate_vp[:, :, np.newaxis]).squeeze(-1)
        endstate_ab = np.matmul(Rpb, endstate_ap[:, :, np.newaxis]).squeeze(-1)

        return np.concatenate((endstate_p, endstate_vb, endstate_ab), axis=1)


    def prepare_input(self, obs):
        """
            Transform the observation to the primitive frame (Body frame → Primitive frame → Body frame).
            obs: [batch; vx, vy, yz, ax, ay, az, gx, gy, gz] in body frame
            :return [batch; vx, vy, yz, ax, ay, az, gx, gy, gz; primitive_v; primitive_h] in primitive frame
            我的理解是计算无人机朝向这个每个prim方向的下的v,vy,vz,ax,ay,az,以及gx,gy,gz,
            是在原有朝向基础上进行每个prim偏移的朝向处理得到9种新的朝向(中间就是保持原本朝向)，然后计算每种朝向下的后续轨迹
        """
        B, N = obs.shape[0], self.lattice_primitive.traj_num

        # 获取所有 Rbp 并倒序排列 (由于lattice和grid的顺序相反)
        Rbp_all = self.lattice_primitive.getRotation().flip(0)  # shape: [N, 3, 3]  # 从右下开始转化为从左上开始

        obs = obs.view(B, 3, 3)  # [B, 3, 3]

        # 扩展 obs 和 Rbp 到 [B, N, 3, 3]
        obs_exp = obs[:, None, :, :].expand(B, N, 3, 3)
        Rbp_exp = Rbp_all[None, :, :, :].expand(B, N, 3, 3)

        # 执行批量坐标变换
        transformed = torch.matmul(obs_exp, Rbp_exp)  # [B, N, 3, 3]

        transformed_flat = transformed.view(B, N, 9)  # [B, N, 9]
        out = transformed_flat.permute(0, 2, 1).contiguous()  # [B, 9, N]
        out = out.view(B, 9, self.lattice_primitive.vertical_num, self.lattice_primitive.horizon_num)  # [B, 9, V, H]
        return out

    def unnormalize_obs(self, vel_acc):
        vel_acc[:, 0:3] = vel_acc[:, 0:3] * self.lattice_primitive.vel_max
        vel_acc[:, 3:6] = vel_acc[:, 3:6] * self.lattice_primitive.acc_max
        return vel_acc

    def normalize_obs(self, vel_acc_goal):
        vel_acc_goal[:, 0:3] = vel_acc_goal[:, 0:3] / self.lattice_primitive.vel_max
        vel_acc_goal[:, 3:6] = vel_acc_goal[:, 3:6] / self.lattice_primitive.acc_max
        vel_acc_goal[:, 6:9] = vel_acc_goal[:, 6:9] / self.goal_length  # 90%为单位向量,10%为小于单位向量的向量

        # Clamp the goal direction to unit length，less than unit length means close to target
        goal_norm_length = vel_acc_goal[:, 6:9].norm(dim=1, keepdim=True)
        scaling = goal_norm_length.clamp(max=1.0) / (goal_norm_length + 1e-8)
        vel_acc_goal[:, 6:9] = vel_acc_goal[:, 6:9] * scaling
        return vel_acc_goal


def rotate_body2world(rot_wb, pos_b):
    """
    Rotate pos_b from body frame to world frame using quaternion q_wb.
    rot_wb: (..., 3, 3)
    pos_b: (..., 3)
    """
    pos_w = torch.matmul(rot_wb, pos_b.unsqueeze(-1)).squeeze(-1)
    return pos_w


def transform_body2world(rot_wb, t_w, pos_b):
    """
    Transform pos_b from body frame to world frame using quaternion q_wb and t_w.
    rot_wb: (..., 3, 3)
    t_w: (..., 3)
    pos_b: (..., 3)
    """
    return rotate_body2world(rot_wb, pos_b) + t_w


def state_body2world(pos_w, rot_wb, pos_b, vel_b, acc_b):
    pos_b = transform_body2world(rot_wb, pos_w, pos_b)
    vel_b = rotate_body2world(rot_wb, vel_b)
    acc_b = rotate_body2world(rot_wb, acc_b)
    return pos_b, vel_b, acc_b


if __name__ == '__main__':
    CoordTransform = StateTransform()
