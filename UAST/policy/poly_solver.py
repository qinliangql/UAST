import numpy as np


class Poly5Solver:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """ 5-th order polynomial at each Axis """
        State_Mat = np.array([pos0, vel0, acc0, pos1, vel1, acc1])
        t = Tf
        Coef_inv = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1 / 2, 0, 0, 0],
                             [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
                             [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
                             [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)]])
        self.A = np.dot(Coef_inv, State_Mat)

    def get_snap(self, t):
        """Return the scalar jerk at time t."""
        return 24 * self.A[4] + 120 * self.A[5] * t

    def get_jerk(self, t):
        """Return the scalar jerk at time t."""
        return 6 * self.A[3] + 24 * self.A[4] * t + 60 * self.A[5] * t * t

    def get_acceleration(self, t):
        """Return the scalar acceleration at time t."""
        return 2 * self.A[2] + 6 * self.A[3] * t + 12 * self.A[4] * t * t + 20 * self.A[5] * t * t * t

    def get_velocity(self, t):
        """Return the scalar velocity at time t."""
        return self.A[1] + 2 * self.A[2] * t + 3 * self.A[3] * t * t + 4 * self.A[4] * t * t * t + \
            5 * self.A[5] * t * t * t * t

    def get_position(self, t):
        """Return the scalar position at time t."""
        return self.A[0] + self.A[1] * t + self.A[2] * t * t + self.A[3] * t * t * t + self.A[4] * t * t * t * t + \
            self.A[5] * t * t * t * t * t


class Polys5Solver:
    def __init__(self, pos0, vel0, acc0, pos1, vel1, acc1, Tf):
        """ multiple 5-th order polynomials at each Axis (only used for visualization of multiple trajectories) """
        N = len(pos1)
        State_Mat = np.array([[pos0] * N, [vel0] * N, [acc0] * N, pos1, vel1, acc1])
        t = Tf
        Coef_inv = np.array([[1, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 1 / 2, 0, 0, 0],
                             [-10 / t ** 3, -6 / t ** 2, -3 / (2 * t), 10 / t ** 3, -4 / t ** 2, 1 / (2 * t)],
                             [15 / t ** 4, 8 / t ** 3, 3 / (2 * t ** 2), -15 / t ** 4, 7 / t ** 3, -1 / t ** 2],
                             [-6 / t ** 5, -3 / t ** 4, -1 / (2 * t ** 3), 6 / t ** 5, -3 / t ** 4, 1 / (2 * t ** 3)]])
        self.A = np.dot(Coef_inv, State_Mat)

    def get_position(self, t):
        """Return the position array at time t."""
        t = np.atleast_1d(t)
        result = (self.A[0][:, np.newaxis] + self.A[1][:, np.newaxis] * t + self.A[2][:, np.newaxis] * t ** 2 +
                  self.A[3][:, np.newaxis] * t ** 3 + self.A[4][:, np.newaxis] * t ** 4 + self.A[5][:, np.newaxis] * t ** 5)
        return result.flatten()


def calculate_yaw(vel_dir, goal_dir, last_yaw, dt, max_yaw_rate=0.3):
    YAW_DOT_MAX_PER_SEC = max_yaw_rate * np.pi
    # Direction of velocity
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-5)

    # Direction of goal
    goal_dist = np.linalg.norm(goal_dir)
    goal_dir = goal_dir / (goal_dist + 1e-5)  # Prevent division by zero

    # Dynamically adjust weights between goal and velocity directions in yaw planning
    goal_yaw = np.arctan2(goal_dir[1], goal_dir[0])
    delta_yaw = goal_yaw - last_yaw
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
    weight = 6 * abs(delta_yaw) / np.pi  #  weight ∈ 6 * [0, 1]  equal weight at 30°, goal weight increases as angle grows

    # Desired direction
    dir_des = vel_dir + weight * goal_dir

    # Temporary yaw calculation
    yaw_temp = np.arctan2(dir_des[1], dir_des[0]) if goal_dist > 0.2 else last_yaw
    max_yaw_change = YAW_DOT_MAX_PER_SEC * dt

    # Logic for yaw adjustment
    if yaw_temp - last_yaw > np.pi:
        if yaw_temp - last_yaw - 2 * np.pi < -max_yaw_change:
            yaw = last_yaw - max_yaw_change
            if yaw < -np.pi:
                yaw += 2 * np.pi
            yawdot = -YAW_DOT_MAX_PER_SEC
        else:
            yaw = yaw_temp
            if yaw - last_yaw > np.pi:
                yawdot = -YAW_DOT_MAX_PER_SEC
            else:
                yawdot = (yaw_temp - last_yaw) / dt
    elif yaw_temp - last_yaw < -np.pi:
        if yaw_temp - last_yaw + 2 * np.pi > max_yaw_change:
            yaw = last_yaw + max_yaw_change
            if yaw > np.pi:
                yaw -= 2 * np.pi
            yawdot = YAW_DOT_MAX_PER_SEC
        else:
            yaw = yaw_temp
            if yaw - last_yaw < -np.pi:
                yawdot = YAW_DOT_MAX_PER_SEC
            else:
                yawdot = (yaw_temp - last_yaw) / dt
    else:
        if yaw_temp - last_yaw < -max_yaw_change:
            yaw = last_yaw - max_yaw_change
            if yaw < -np.pi:
                yaw += 2 * np.pi
            yawdot = -YAW_DOT_MAX_PER_SEC
        elif yaw_temp - last_yaw > max_yaw_change:
            yaw = last_yaw + max_yaw_change
            if yaw > np.pi:
                yaw -= 2 * np.pi
            yawdot = YAW_DOT_MAX_PER_SEC
        else:
            yaw = yaw_temp
            if yaw - last_yaw > np.pi:
                yawdot = -YAW_DOT_MAX_PER_SEC
            elif yaw - last_yaw < -np.pi:
                yawdot = YAW_DOT_MAX_PER_SEC
            else:
                yawdot = (yaw_temp - last_yaw) / dt

    return yaw, yawdot


# A modified version of calculate_yaw to prioritize goal direction more strongly
def calculate_yaw_track(vel_dir, goal_dir, last_yaw, dt, max_yaw_rate=0.3):
    YAW_DOT_MAX_PER_SEC = max_yaw_rate * np.pi
    # Direction of velocity
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-5)

    # Direction of goal
    goal_dist = np.linalg.norm(goal_dir)
    goal_dir = goal_dir / (goal_dist + 1e-5)  # Prevent division by zero

    # Dynamically adjust weights between goal and velocity directions in yaw planning
    goal_yaw = np.arctan2(goal_dir[1], goal_dir[0])
    delta_yaw = goal_yaw - last_yaw
    delta_yaw = (delta_yaw + np.pi) % (2 * np.pi) - np.pi  # wrap to [-π, π]
    # 修改：设置固定高权重确保目标方向优先
    weight = 10.0  # 固定高权重，使目标方向主导

    # Desired direction
    # 修改：目标方向权重显著提高
    dir_des = 0.1 * vel_dir + weight * goal_dir

    # Temporary yaw calculation
    # 修改：移除近距离条件限制，始终跟踪目标方向
    yaw_temp = np.arctan2(dir_des[1], dir_des[0])
    max_yaw_change = YAW_DOT_MAX_PER_SEC * dt

    # Logic for yaw adjustment
    if yaw_temp - last_yaw > np.pi:
        if yaw_temp - last_yaw - 2 * np.pi < -max_yaw_change:
            yaw = last_yaw - max_yaw_change
            if yaw < -np.pi:
                yaw += 2 * np.pi
            yawdot = -YAW_DOT_MAX_PER_SEC
        else:
            yaw = yaw_temp
            if yaw - last_yaw > np.pi:
                yawdot = -YAW_DOT_MAX_PER_SEC
            else:
                yawdot = (yaw_temp - last_yaw) / dt
    elif yaw_temp - last_yaw < -np.pi:
        if yaw_temp - last_yaw + 2 * np.pi > max_yaw_change:
            yaw = last_yaw + max_yaw_change
            if yaw > np.pi:
                yaw -= 2 * np.pi
            yawdot = YAW_DOT_MAX_PER_SEC
        else:
            yaw = yaw_temp
            if yaw - last_yaw < -np.pi:
                yawdot = YAW_DOT_MAX_PER_SEC
            else:
                yawdot = (yaw_temp - last_yaw) / dt
    else:
        if yaw_temp - last_yaw < -max_yaw_change:
            yaw = last_yaw - max_yaw_change
            if yaw < -np.pi:
                yaw += 2 * np.pi
            yawdot = -YAW_DOT_MAX_PER_SEC
        elif yaw_temp - last_yaw > max_yaw_change:
            yaw = last_yaw + max_yaw_change
            if yaw > np.pi:
                yaw -= 2 * np.pi
            yawdot = YAW_DOT_MAX_PER_SEC
        else:
            yaw = yaw_temp
            if yaw - last_yaw > np.pi:
                yawdot = -YAW_DOT_MAX_PER_SEC
            elif yaw - last_yaw < -np.pi:
                yawdot = YAW_DOT_MAX_PER_SEC
            else:
                yawdot = (yaw_temp - last_yaw) / dt

    return yaw, yawdot