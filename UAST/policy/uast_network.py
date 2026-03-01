"""
UAST Network
forward, prediction, pre-processing, post-processing
"""

import torch
from torch import nn
import numpy as np
from policy.models.backbone import YopoBackbone, MaskBackbone
from policy.models.head import YopoHead
from policy.state_transform import *


class YopoNetwork(nn.Module):

    def __init__(
            self,
            observation_dim=9,  # 9: v_xyz, a_xyz, goal_xyz
            output_dim=10,  # 10: x_pva, y_pva, z_pva, score # 这里好像写错了，应该是pxyz,vxyz,axyz,score
            hidden_state=64,
    ):
        super(YopoNetwork, self).__init__()
        self.state_transform = StateTransform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_backbone = YopoBackbone(hidden_state)
        self.state_backbone = nn.Sequential()   # do nothing, just for compatibility
        self.yopo_head = YopoHead(hidden_state + observation_dim, output_dim)

    def forward(self, depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
            forward propagation of neural network
        """
        depth_feature = self.image_backbone(depth)  # [B, 64, 3, 5]
        obs_feature = self.state_backbone(obs)  # [B, 9, 3, 5], obs: [B, 9, vertical_num, horizon_num]
        input_tensor = torch.cat((obs_feature, depth_feature), 1)
        output = self.yopo_head(input_tensor)   # [B, 10, vertical_num, horizon_num]
        endstate = torch.tanh(output[:, :9])  # [batch, 9, vertical_num, horizon_num], process to [-1,1]
        score = torch.nn.functional.softplus(output[:, 9])  # [batch, vertical_num, horizon_num], process to [0, +inf]
        return endstate, score

    def inference(self, depth: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
            For network training:
            (1) normalize the input state and transform to primitive frame
            (2) forward propagation
            (3) convert the prediction to endstate in body frame.
            obs: current state in the body frame.
            return: end state in the body frame
        """
        obs = self.state_transform.normalize_obs(obs)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred = self.forward(depth, obs)
        endstate = self.state_transform.pred_to_endstate(endstate_pred)
        return endstate, score_pred

    def print_grad(self, grad):
        print("grad of hook: ", grad)

class UASTNetworkTrack(nn.Module):

    def __init__(
            self,
            observation_dim=9,  # 9: v_xyz, a_xyz, goal_xyz
            output_dim=10,  # 10: x_pva, y_pva, z_pva, score # 这里好像写错了，应该是pxyz,vxyz,axyz,score
            hidden_state=64,
    ):
        super(UASTNetworkTrack, self).__init__()
        self.state_transform = StateTransform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_backbone = YopoBackbone(hidden_state)
        self.mask_backbone = MaskBackbone(hidden_state//8)
        self.state_backbone = nn.Sequential()   # do nothing, just for compatibility
        self.yopo_head = YopoHead(hidden_state + hidden_state//8 + observation_dim, output_dim)

    def forward(self, depth: torch.Tensor, mask: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
            forward propagation of neural network
        """
        depth_feature = self.image_backbone(depth)  # [B, 64, 3, 5]
        mask_feature = self.mask_backbone(mask)  # [B, 8, 3, 5]
        obs_feature = self.state_backbone(obs)  # [B, 9, 3, 5], obs: [B, 9, vertical_num, horizon_num]
        input_tensor = torch.cat((obs_feature, depth_feature, mask_feature), 1)
        output = self.yopo_head(input_tensor)   # [B, 10, vertical_num, horizon_num]
        endstate = torch.tanh(output[:, :9])  # [batch, 9, vertical_num, horizon_num], process to [-1,1]
        score = torch.nn.functional.softplus(output[:, 9])  # [batch, vertical_num, horizon_num], process to [0, +inf]
        return endstate, score

    def inference(self, depth: torch.Tensor, mask: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """
            For network training:
            (1) normalize the input state and transform to primitive frame
            (2) forward propagation
            (3) convert the prediction to endstate in body frame.
            obs: current state in the body frame.
            return: end state in the body frame
        """
        obs = self.state_transform.normalize_obs(obs)
        obs = self.state_transform.prepare_input(obs)
        endstate_pred, score_pred = self.forward(depth, mask, obs)
        endstate = self.state_transform.pred_to_endstate(endstate_pred)
        return endstate, score_pred

    def print_grad(self, grad):
        print("grad of hook: ", grad)
