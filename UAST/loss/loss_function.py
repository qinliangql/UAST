import os
import math
import torch as th
import torch.nn as nn
from ruamel.yaml import YAML
from loss.safety_loss import SafetyLoss
from loss.smoothness_loss import SmoothnessLoss
from loss.guidance_loss import GuidanceLoss
from loss.track_loss import TrackLoss


class UASTLoss(nn.Module):
    def __init__(self):
        """
        Compute the cost: including smoothness, safety, guidance, goal cost, etc.
        Currently, keeping multi-segment polynomial support (not yet verified), but only using a single-segment polynomial (m = 1) for now.
        dp: decision parameters
        df: fixed parameters
        """
        super(UASTLoss, self).__init__()
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.cfg = YAML().load(open(os.path.join(base_dir, "../config/traj_opt.yaml"), 'r'))
        self.sgm_time = 2 * self.cfg["radio_range"] / self.cfg["velocity"]
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self._C, self._B, self._L, self._R = self.qp_generation()
        self._R = self._R.to(self.device)
        self._L = self._L.to(self.device)
        self.smoothness_weight = self.cfg["ws"]
        self.safety_weight = self.cfg["wc"]
        self.goal_weight = self.cfg["wg"]
        self.track_weight = self.cfg["wt"]
        self.smoothness_loss = SmoothnessLoss(self._R)
        self.safety_loss = SafetyLoss(self._L, self.sgm_time)
        self.goal_loss = GuidanceLoss()
        self.track_loss = TrackLoss(self._L, self.sgm_time)
        print("---------- Loss ---------")
        print(f"| {'smooth':<12} = {self.smoothness_weight:6.4f} |")
        print(f"| {'safety':<12} = {self.safety_weight:6.4f} |")
        print(f"| {'goal':<12} = {self.goal_weight:6.4f} |")
        print("-------------------------")

    def qp_generation(self):
        # 论文中的映射矩阵，将多项式的系数与轨迹的边界条件（位置、速度、加速度）关联起来
        A = th.zeros((6, 6))
        for i in range(3):
            A[2 * i, i] = math.factorial(i)
            for j in range(i, 6):
                A[2 * i + 1, j] = math.factorial(j) / math.factorial(j - i) * (self.sgm_time ** (j - i))

        # H海森矩阵，论文中的矩阵Q
        # 在二次规划问题中用于衡量轨迹的平滑性。它决定了轨迹多项式系数的二次项权重，通过最小化与 H 相关的二次型可以得到平滑的轨迹。
        H = th.zeros((6, 6))
        for i in range(3, 6):
            for j in range(3, 6):
                H[i, j] = i * (i - 1) * (i - 2) * j * (j - 1) * (j - 2) / (i + j - 5) * (self.sgm_time ** (i + j - 5))

        return self.stack_opt_dep(A, H)

    def stack_opt_dep(self, A, Q):
        Ct = th.zeros((6, 6))
        Ct[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1

        _C = th.transpose(Ct, 0, 1) # 置换矩阵，用于调整矩阵元素的顺序

        B = th.inverse(A)   # 用于将轨迹的边界条件转换为多项式系数

        B_T = th.transpose(B, 0, 1)

        _L = B @ Ct

        _R = _C @ (B_T) @ Q @ B @ Ct

        return _C, B, _L, _R

    def forward(self, state, prediction, goal, map_id):
        """
        Args:
            prediction: (batch_size, 3, 3) → [px, py, pz; vx, vy, vz; ax, ay, az] in world frame [B*V*H, 3, 3]
            state: (batch_size, 3, 3) → [px, py, pz; vx, vy, vz; ax, ay, az] in world frame [B*V*H, 3, 3]
            goal: (batch_size, 3) → (px, py, pz) in world frame    [B*V*H, 3]
            map_id: (batch_size) which ESDF map to query    [B]

        Returns:
            cost: (batch_size) → weighted cost
        """
        # Fixed part: initial pos, vel, acc → (batch_size, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Df = state.permute(0, 2, 1)

        # Decision parameters (local frame) → (batch_size, 3, 3) [px, vx, ax; py, vy, ay; pz, vz, az]
        Dp = prediction.permute(0, 2, 1)

        smoothness_cost = th.tensor(0.0, device=self.device, requires_grad=True)
        safety_cost = th.tensor(0.0, device=self.device, requires_grad=True)
        goal_cost = th.tensor(0.0, device=self.device, requires_grad=True)
        track_cost = th.tensor(0.0, device=self.device, requires_grad=True)

        if self.smoothness_weight > 0:
            smoothness_cost = self.smoothness_loss(Df, Dp)
        if self.safety_weight > 0:
            safety_cost = self.safety_loss(Df, Dp, map_id)
        if self.goal_weight > 0:
            goal_cost = self.goal_loss(Df, Dp, goal)
        if self.track_weight > 0:
            track_cost = self.track_loss(Df, Dp, map_id, goal)

        return self.smoothness_weight * smoothness_cost, self.safety_weight * safety_cost, \
            self.goal_weight * goal_cost, self.track_weight * track_cost