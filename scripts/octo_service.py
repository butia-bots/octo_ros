from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, RHCWrapper, UnnormalizeActionProprio
from interbotix_xs_modules.arm import InterbotixManipulatorXS
import gym
import rospy
from sensor_msgs.msg import Image
from ros_numpy import numpify
import cv2
import numpy as np
from tf.transformations import euler_from_matrix, euler_matrix

class InterbotixGym(gym.Env):
    def __init__(self, img_size=256, sticky_gripper_num_steps=1):
        self.bot = InterbotixManipulatorXS("doris_arm", "arm", "gripper")
        self.img_sub = rospy.Subscriber("/camera/image_color/raw", Image, callback=self._on_images)
        self.img_size = img_size
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=np.zeros((img_size, img_size, 3)),
                    high=255 * np.ones((img_size, img_size, 3)),
                    dtype=np.uint8,
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.zeros((7,)), high=np.ones((7,)), dtype=np.float64
        )
        self.sticky_gripper_num_steps = sticky_gripper_num_steps
        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

    def step(self, action):
        # sticky gripper logic
        if (action[-1] < 0.5) != self.is_gripper_closed:
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.is_gripper_closed = not self.is_gripper_closed
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.0 if self.is_gripper_closed else 1.0
        action[:6] -= 0.5
        dx, dy, dz, dyaw, dpitch, droll = action[:6]
        delta = np.array([dx, dy, dz, droll, dpitch, dyaw])
        pose_matrix = self.bot.arm.get_ee_pose(self)
        position = pose_matrix[:3,3].flatten()
        orientation = pose_matrix[:3,:3]
        orientation = euler_from_matrix(orientation)
        pose = np.concatenate([position, orientation])
        target_pose = pose + delta
        x, y, z, roll, pitch, yaw = target_pose
        yaw = None
        self.bot.arm.set_ee_pose_components(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw)
        if action[-1] == 0.0:
            self.bot.gripper.close()
        else:
            self.bot.gripper.open()
        obs = self._get_obs()
        return obs, 0, False, truncated, {}

    def reset(self):
        #TODO: reset arm to a pose that is visible to the camera
        self.bot.gripper.open()
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "image_primary": cv2.resize(numpify(self._img), (self.img_size, self.img_size))
        }

    def _on_images(self, msg):
        self._img = msg

if __name__ == '__main__':
    model = OctoModel.load_pretrained("hf://rail-berkeley/octo-base")
    print(model.get_pretty_spec())
    env = InterbotixGym()
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=50)
    env = UnnormalizeActionProprio(env, model.dataset_statistics, normalization_type='normal')
