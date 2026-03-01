import torch
from scipy.spatial.transform import Rotation as R


class LatticeParam:
    def __init__(self, cfg):
        ratio = cfg["velocity"] / cfg["vel_align"]
        self.vel_max = ratio * cfg["vel_align"]
        self.acc_max = ratio * ratio * cfg["acc_align"]
        self.segment_time = 2 * cfg["radio_range"] / self.vel_max   # 多久走完到目标
        self.horizon_num = cfg["horizon_num"]
        self.vertical_num = cfg["vertical_num"]
        self.radio_num = cfg["radio_num"]
        self.horizon_fov = cfg["horizon_camera_fov"]
        self.vertical_fov = cfg["vertical_camera_fov"]
        self.horizon_anchor_fov = cfg["horizon_anchor_fov"]
        self.vertical_anchor_fov = cfg["vertical_anchor_fov"]
        self.radio_range = cfg["radio_range"]

        print("---------- Param --------")
        print(f"| {'max speed':<12} = {round(self.vel_max, 1):>6} |")
        print(f"| {'max accel':<12} = {round(self.acc_max, 1):>6} |")
        print(f"| {'traj time':<12} = {round(self.segment_time, 1):>6} |")
        print(f"| {'max radio':<12} = {round(2 * self.radio_range, 1):>6} |")
        print("-------------------------")


class LatticePrimitive(LatticeParam):
    """
    Grid index layout in image (Polar coordinate indexing: row-major, bottom-left origin):
                       +---+---+---+
                       | 8 | 7 | 6 |
                       +---+---+---+
                       | 5 | 4 | 3 |
                       +---+---+---+
                       | 2 | 1 | 0 |
                       +---+---+---+
    """
    _instance = None

    def __init__(self, cfg):
        super().__init__(cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.traj_num = self.vertical_num * self.horizon_num * self.radio_num

        if self.horizon_num == 1:
            direction_diff = 0
        else:
            direction_diff = (self.horizon_fov / 180.0 * torch.pi) / self.horizon_num

        if self.vertical_num == 1:
            altitude_diff = 0
        else:
            altitude_diff = (self.vertical_fov / 180.0 * torch.pi) / self.vertical_num
        radio_diff = self.radio_range / self.radio_num

        lattice_pos_list = []
        lattice_angle_list = []
        lattice_Rbp_list = []

        # Primitives: Bottom to Top, Right to Left, Shot to Long(un-used)
        for h in range(0, self.radio_num):
            for i in range(0, self.vertical_num):
                for j in range(0, self.horizon_num):
                    search_radio = (h + 1) * radio_diff
                    alpha = torch.tensor(-direction_diff * (self.horizon_num - 1) / 2 + j * direction_diff)
                    beta = torch.tensor(-altitude_diff * (self.vertical_num - 1) / 2 + i * altitude_diff)

                    pos_node = torch.tensor([torch.cos(beta) * torch.cos(alpha) * search_radio,
                                             torch.cos(beta) * torch.sin(alpha) * search_radio,
                                             torch.sin(beta) * search_radio])

                    lattice_pos_list.append(pos_node)
                    lattice_angle_list.append(torch.tensor([alpha, beta]))
                    Rotation = R.from_euler('ZYX', [alpha, -beta, 0.0], degrees=False)  # inner rotation: yaw-pitch-roll
                    lattice_Rbp_list.append(torch.tensor(Rotation.as_matrix()))

        self.lattice_pos_node = torch.stack(lattice_pos_list).to(dtype=torch.float32, device=device)  # shape: [N, 3]
        self.lattice_angle_node = torch.stack(lattice_angle_list).to(dtype=torch.float32, device=device)  # shape: [N, 2]
        self.lattice_Rbp_node = torch.stack(lattice_Rbp_list).to(dtype=torch.float32, device=device)  # shape: [N, 3, 3]

        self.yaw_diff = 0.5 * self.horizon_anchor_fov / 180.0 * torch.pi
        self.pitch_diff = 0.5 * self.vertical_anchor_fov / 180.0 * torch.pi

    def getStateLattice(self, id=None):
        if id is not None:
            return self.lattice_pos_node[id, :]
        else:
            return self.lattice_pos_node

    def getAngleLattice(self, id=None):
        if id is not None:
            return self.lattice_angle_node[id, 0], self.lattice_angle_node[id, 1]  # yaw, pitch
        else:
            return self.lattice_angle_node[:, 0], self.lattice_angle_node[:, 1]  # yaw, pitch

    def getRotation(self, id=None):
        if id is not None:
            return self.lattice_Rbp_node[id]
        else:
            return self.lattice_Rbp_node

    def convert_ImageGrid_LatticeID(self, id):
        return self.traj_num - id - 1

    @classmethod
    def get_instance(self, cfg):
        if self._instance is None: 
            self._instance = self(cfg)
        return self._instance


if __name__ == '__main__':
    import os
    from ruamel.yaml import YAML

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = YAML().load(open(os.path.join(base_dir, "../config/traj_opt.yaml"), 'r'))
    lattice_primitive = LatticePrimitive.get_instance(cfg)
    print(lattice_primitive.getStateLattice(list(range(lattice_primitive.traj_num))))
