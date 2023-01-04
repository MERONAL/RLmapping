import time
import math
import numpy as np
import rospy
from gym import spaces
from gym.utils import seeding


from geometry_msgs.msg import Twist, Point
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
from amr_robots_gazebo.srv import Planning

from rl.scripts.envs.gazebo_env import GazeboEnv
from rl.scripts.envs.assets.data.static_object_env_data import get_random_target, is_collision


class LocalNavEnv(GazeboEnv):
    """
    static env for local obstacle avoidance.
    """
    def __init__(self, launchfile) -> None:
        super().__init__(launchfile)
        # publisher
        self.vel_pub = rospy.Publisher("/sim_p3at/cmd_vel", Twist, queue_size=5)
        # sub
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, callback=self._scan_cb)
        self.pose_sub = rospy.Subscriber(
            "/sim_p3at/odom", Odometry, callback=self._odom_cb
        )
        # topic data
        self._odom_msg = None
        self._scan_data = None

        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.set_gazebo = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

        self.info = {}
        self.obs_dim = 16
        self.observation_space = spaces.Box(-np.inf, inf, shape=(self.obs_dim,))
        self.action_space = spaces.Discrete(3)  # F, L, R
        self.reward_range = (-np.inf, np.inf)
        self._seed()

        self.local_target = None
        self.global_target = None

        self.max_done = False

        self.MAX_STEP = 512
        self.current_step = 0  # this variable indicates for each episode, we need to confirm that current_step < MAX_STEP.

        self.total_step = 0  # use this variable for curriclum learning.
        self.target_distance_gap = 10000
        self.CUR_TARGET_DIS = 40
        self._MIN_DISTANCE = 8
        self._SIMULATION_TIMES = 0
        self._MINI_PATH_LENGTH = 0
        self._REAL_PATH_LENGTH = 0
        self.trajectories = {"position": [], "start_pos": [0, 0], "target_pos": [0, 0]}

    def eval_model(self):
        self.CUR_TARGET_DIS = 1000
        self._SIMULATION_TIMES = 0

    def _scan_cb(self, data: LaserScan) -> None:
        """laser scan callback"""
        self._scan_data = data

    def _odom_cb(self, data: Odometry) -> None:
        """odommetry callback, get the pose of robots"""
        self._odom_msg = data.pose.pose

    @staticmethod
    def _local_target(global_target: np.ndarray) -> np.ndarray:
        """local navigation target"""
        data = Planning()
        data.x = global_target[0]
        data.y = global_target[1]
        rospy.wait_for_service("/local_target", 5)
        try:
            get_local_target = rospy.ServiceProxy("/local_target", Planning)
            res = get_local_target(100, 0)

            return np.array([res.x, res.y])
        except rospy.ServiceException as e:
            print("[LocalNavigation][ERROR] ", e)

    def _is_collision(self) -> bool:
        """judge whether is collision"""
        pos_x = self._odom_msg.position.x
        pos_y = self._odom_msg.position.y
        return is_collision(pos_x=pos_x, pos_y=pos_y)

    def distance(self, cur):
        dis = np.linalg.norm(self.local_target - cur)
        return dis

    def _seed(self, seed: int = None):
        self.np_random, seed = seeding.np_random(seed=seed)
        return [seed]

    @staticmethod
    def discretize_observation(data, new_ranges):
        discretized_ranges = []
        mod = len(data.ranges) / new_ranges
        for i, item in enumerate(data.ranges):
            if i % mod == 0:
                if data.ranges[i] == float("Inf"):
                    discretized_ranges.append(30.0)
                elif np.isnan(data.ranges[i]):
                    discretized_ranges.append(0.0)
                else:
                    discretized_ranges.append(data.ranges[i])
        return np.array(discretized_ranges)

    def step(self, action):
        self.current_step += 1
        self.total_step += 1
        # pause the simulation engine
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/unpause_physics service call failed")

        # get sensor data.
        vel_cmd = self._scale_action_to_control(action)
        self.vel_pub.publish(vel_cmd)  # step
        # get data.
        self.ros_sleep()
        is_collision = self._is_collision()
        position = np.array([self._odom_msg.position.x, self._odom_msg.position.y])
        orientation = np.array(
            [
                self._odom_msg.orientation.x,
                self._odom_msg.orientation.y,
                self._odom_msg.orientation.z,
                self._odom_msg.orientation.w,
            ]
        )
        laser_data = self._scan_data
        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/pause_physics service call failed")

        laser_data = self.discretize_observation(laser_data, 10)
        pos_state = self.get_pos_state(position, self.local_target)
        state = np.concatenate((pos_state, orientation, laser_data))
        done = is_collision
        reward, done = self._get_reward(position, is_collision, action)
        self.max_done = False
        # maximum steps.
        if self.current_step >= self.MAX_STEP:
            reward = -500
            done = True
            self.max_done = True
        self.clear_data()
        info = {}
        global_done = False  # indicate whether get the global target location.
        self.trajectories["position"].append(position)
        if done:
            global_done = self.__global_target_done(position, self.global_target)
            if global_done:
                print(
                    f"\033[1;31m[GlobalNavigation][INFO] success navigate to global target:[{self.global_target[0]}, {self.global_target[1]}]\033[0m"
                )
            info["path_length"] = self._get_path_length()
            info["mini_path"] = math.sqrt(
                (self.trajectories["start_pos"][0] - self.trajectories["target_pos"][0])
                ** 2
                + (
                    self.trajectories["start_pos"][1]
                    - self.trajectories["target_pos"][1]
                )
                ** 2
            )
        info["global_done"] = global_done
        return state, reward, done, info

    def _get_reward(self, cur, is_collision, action):
        if is_collision:
            print(f"[LocalNavigation][INFO] occur collision")
            return -100, True
        current_distance = self.distance(cur)
        # if success
        if current_distance < 0.5:
            print(
                f"[LocalNavigation][INFO] successful arrival local target:[{self.local_target[0]}, {self.local_target[1]}]"
            )
            return 1000, True

        reward = -0.1  # each step
        reward += action[0] * np.cos(action[1]) * 0.5
        return reward, False

    def reset(self):
        # # resets the state of the environment and returns an initial observation
        # rospy.wait_for_service('/gazebo/reset_simulation')

        # try:
        #     self.reset_proxy()
        # except rospy.ServiceException as e:
        #     rospy.logerr("/gazebo/reset_simulation service call failed.")
        # # Unpause simulation to make observation
        rospy.wait_for_service("/gazebo/unpause_physics")

        try:
            self.unpause()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/unpause_physics service call failed")
        self.ros_sleep()
        position = np.array([self._odom_msg.position.x, self._odom_msg.position.y])
        if (
            self.global_target is None
            or self.__global_target_done(position, self.global_target)
            or self._is_collision()
            or self.max_done
        ):
            self._SIMULATION_TIMES += 1
            self.__reset_robot()
            self.ros_sleep()
            position = np.array([self._odom_msg.position.x, self._odom_msg.position.y])
            self.global_target = get_random_target(
                position,
                self.CUR_TARGET_DIS + self.total_step / self.target_distance_gap,
                self._MIN_DISTANCE,
            )

            # record the trajectories.
            self.trajectories["position"] = []
            self.trajectories["start_pos"] = position
            self.trajectories["target_pos"] = self.global_target

        self.local_target = self._local_target(self.global_target)
        print(
            f"\033[1;34m[LocalNavigation][INFO] current pos:[{self.local_target[0]}, {self.local_target[1]}], global target:[{self.global_target[0]}, {self.global_target[1]}]\033[0m"
        )
        orientation = np.array(
            [
                self._odom_msg.orientation.x,
                self._odom_msg.orientation.y,
                self._odom_msg.orientation.z,
                self._odom_msg.orientation.w,
            ]
        )
        laser_data = self._scan_data

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except rospy.ServiceException as e:
            rospy.logerr("/gazebo/pause_physics service call failed")
        laser_data = self.discretize_observation(laser_data, 10)  # laser_data
        pos_state = self.get_pos_state(position, self.local_target)
        state = np.concatenate((pos_state, orientation, laser_data))
        self.current_step = 0
        self.clear_data()
        return state

    @staticmethod
    def __global_target_done(
        current_pos: np.ndarray, global_target: np.ndarray
    ) -> bool:
        dis = np.linalg.norm(current_pos - global_target)
        if dis < 2:
            return True
        return False

    def clear_data(self):
        self._odom_msg = None
        self._scan_data = None

    def ros_sleep(self):
        while not rospy.core.is_shutdown() and (
            self._odom_msg is None or self._scan_data is None
        ):
            rospy.rostime.wallsleep(0.01)

    @staticmethod
    def get_pos_state(current_position, target_position):
        dis = np.linalg.norm(current_position - target_position)
        angle = np.arctan2(
            target_position[1] - current_position[1],
            target_position[0] - current_position[0],
        )
        return np.array([dis, angle])

    def __reset_robot(self):
        cmd_vel = Twist()
        self.vel_pub.publish(cmd_vel)
        msg = ModelState()
        msg.model_name = "pioneer3at_robot"
        tmp = get_random_target()
        # tmp = [0, 0]
        msg.pose.position.x = tmp[0]
        msg.pose.position.y = tmp[1]
        msg.pose.orientation.z = np.random.rand(1)[0]
        msg.pose.orientation.w = np.random.rand(1)[0]
        while True:
            try:
                self.set_gazebo(msg)
            except rospy.ServiceException as e:
                time.sleep(0.1)
            else:
                break

    def _scale_action_to_control(self, action) -> Twist:
        """
        action = int, Discrete(), [0, 1, 2]
        """
        vel_cmd = Twist()
        if action == 0:  # Forward
            vel_cmd.linear.x = 0.3
            vel_cmd.angular.z = 0.0
        elif action == 1:  # Left
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = 0.3
        elif action == 2:
            vel_cmd.linear.x = 0.05
            vel_cmd.angular.z = -0.3
        return vel_cmd

    def simulation_times(self):
        return self._SIMULATION_TIMES

    def _get_path_length(self):
        "return the trajectories path length for current algorithm."
        pos_data = self.trajectories["position"]
        total_length = 0
        previous_pos = pos_data[0]
        for i in range(1, len(pos_data)):
            dis = math.sqrt(
                (previous_pos[0] - pos_data[i][0]) ** 2
                + (previous_pos[1] - pos_data[i][1]) ** 2
            )
            total_length += dis
            previous_pos = pos_data[i]
        return total_length


class LocalNavContinusEnv(LocalNavEnv):
    def __init__(self) -> None:
        super().__init__()
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)

    def _scale_action_to_control(self, action) -> Twist:
        """
        action = np.array([a, b])
        a: cmd_vel, range:(-1, 1) mapping to (0, 0.3)
        b: angle: range:(-1, 1), mapping to (-0.3, 0.3)
        """
        action[0] = (action[0] + 1) * 0.15  # cmd_val
        action[1] = action[1] * 0.3
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]  # cmd_vel
        vel_cmd.angular.z = action[1]  # angle
        return vel_cmd


class LocalNavContinusEnv2(LocalNavContinusEnv):
    """
    incremental action space.
    """

    def __init__(self) -> None:
        super().__init__()
        self.previous_velocity = np.array([0, 0])  # cmd_vel, angle

    def _scale_action_to_control(self, action) -> Twist:
        """small change based on previous action"""
        action[0] = action[0] * 0.1 + self.previous_velocity[0]  # cmd_val
        action[1] = action[1] * 0.1 + self.previous_velocity[1]
        if action[0] > 0.3:
            action[0] = 0.3
        if action[0] < 0:
            action[0] = 0
        # clip the range
        if action[1] > 0.3:
            action[1] = 0.3
        if action[1] < -0.3:
            action[1] = -0.3
        self.previous_velocity = np.array([action[0], action[1]])
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]  # cmd_vel
        vel_cmd.angular.z = action[1]
        return vel_cmd

    def reset(self):
        self.previous_velocity = np.array([0, 0])
        return super().reset()
