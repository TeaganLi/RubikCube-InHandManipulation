#!/usr/bin/env python
'''
Env for layer-wise operation
'''
import numpy as np
import pickle, copy

from gym import utils
from gym import error, spaces
from gym.envs.robotics import rotations

from manipulation.utils.util import shadow_get_obs, quat_from_angle_and_axis
from manipulation.envs.manipulate_base_env import ManipulateEnv

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))

class CubeRotEnv(ManipulateEnv, utils.EzPickle, ):
    def __init__(self, model_path='rubik_cube_shadow_w_target.xml',target_position='fixed', target_rotation='parallel',
                 randomize_initial_position=True, randomize_initial_rotation=True,
                 rotation_threshold=0.2, distance_threshold=0.02):
        """
        Initializes a new ManipulateCubeEnv environment for cube rotation operation.

        Args:
            model_path (string): path to the environment XML file
            target_position (string): the type of target porisiton:
                - ignore: target position is fully ignored, i.e. the object can be positioned arbitrarily
                - fixed: target position is set to the initial position of the target
                - random: target position is fully randomized according to target_postion_range
            target_rotation (string): the type of target rotation:
                - ignore: target rotation is fully ignored, i.e. the object can be rotated arbitrarily
                - fixed: target rotation is set to the initial rotation of the object
                - xyz: fully randomized target rotation around the X, Y and Z axis
                - z: fully randomized target rotation around the Z axis
                - parallel: fully randomized target rotation around Z and axis-aligned rotation around X, Y
            randomize_initial_position (boolean): whether or not to randomize the initial position of the object
            randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
            distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
            rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered achieved
        """
        self.goal = np.zeros(7)  # target position, orientation
        ManipulateEnv.__init__(self, target_position=target_position,
                               target_rotation=target_rotation, model_path=model_path,
                               randomize_initial_position=randomize_initial_position,
                               randomize_initial_rotation=randomize_initial_rotation,
                               rotation_threshold=rotation_threshold, distance_threshold=distance_threshold)
        self.goal = self._sample_goal()

        utils.EzPickle.__init__(self)

    def _get_obs(self):
        """
        hand joint angle (24), hand joint vel (24), cube center vel (6),
        cube center position (3) + rotation (4)
        total 61
        """
        robot_qpos, robot_qvel = shadow_get_obs(self.sim, name='robot0')
        achieved_goal = self._get_achieved_goal()
        cube_vel = self.sim.data.get_joint_qvel('rubik:free_joint_0_0_0').copy()
        observation = np.concatenate([robot_qpos, robot_qvel, cube_vel, achieved_goal])
        assert observation.shape == (61,)
        return {
            'observation': observation.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _get_achieved_goal(self):
        """
        Cube center position (3), cube center rotation (4)
        """
        # Cube center position and rotation (quaternion).
        cube_qpos = self.sim.data.get_site_xpos('rubik:cube_center')
        cube_rot = rotations.mat2quat(self.sim.data.get_site_xmat('rubik:cube_center'))
        cube_qpos = np.concatenate([cube_qpos, cube_rot])
        assert cube_qpos.shape == (7,)
        return cube_qpos

    def _goal_distance(self, goal_a, goal_b):
        """
        return: position distance and angle difference.
        """
        assert goal_a.shape == goal_b.shape
        assert goal_a.shape[-1] == 7

        d_pos = np.zeros_like(goal_a[..., 0])
        d_rot = np.zeros_like(goal_b[..., 0])
        if self.target_position != 'ignore':
            # delta_pos = goal_a[..., :3] - goal_b[..., :3]
            delta_pos = goal_a[..., :2] - goal_b[..., :2]  # ignore the z distance
            d_pos = np.linalg.norm(delta_pos, axis=-1)

        if self.target_rotation != 'ignore':
            quat_a, quat_b = goal_a[..., 3:], goal_b[..., 3:]
            # Subtract quaternions and extract angle between them
            quat_diff = rotations.quat_mul(quat_a, rotations.quat_conjugate(quat_b))
            angle_diff = 2 * np.arccos(np.clip(quat_diff[..., 0], -1., 1.))
            d_rot = angle_diff
        assert d_pos.shape == d_rot.shape
        return d_pos, d_rot

    def _is_success(self, achieved_goal, desired_goal):
        d_pos, d_rot = self._goal_distance(achieved_goal, desired_goal)
        achieved_pos = (d_pos < self.distance_threshold).astype(np.float32)
        achieved_rot = (d_rot < self.rotation_threshold).astype(np.float32)
        cube_center_pos = self.sim.data.get_site_xpos('rubik:cube_center')
        achieved_height = (cube_center_pos[2] > 0.1).astype(np.float32)
        achieved_both = achieved_pos * achieved_rot * achieved_height
        return achieved_both

    def _sample_goal(self):
        # Select a goal for the object position.
        if self.target_position == 'random':
            assert self.target_position_range.shape == (3, 2)
            offset = self.np_random.uniform(self.target_position_range[:, 0], self.target_position_range[:, 1])
            assert offset.shape == (3,)
            target_pos = self.sim.data.get_site_xpos('rubik:cube_center') + offset
        elif self.target_position in ['ignore', 'fixed']:
            # target_pos = self.sim.data.get_site_xpos('rubik:cube_center')
            target_pos = np.array([1, 0.87, 0.2])
        else:
            raise error.Error('Unknown target_position option "{}".'.format(self.target_position))
        assert target_pos.shape == (3,)

        # Select a goal for the object rotation
        if self.target_rotation == 'z':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation == 'parallel':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = np.array([0., 0., 1.])
            target_quat = quat_from_angle_and_axis(angle, axis)
            parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
            target_quat = rotations.quat_mul(target_quat, parallel_quat)
        elif self.target_rotation == 'xyz':
            angle = self.np_random.uniform(-np.pi, np.pi)
            axis = self.np_random.uniform(-1., 1., size=3)
            target_quat = quat_from_angle_and_axis(angle, axis)
        elif self.target_rotation in ['ignore', 'fixed']:
            target_quat = self.sim.data.get_joint_qpos('rubik:free_joint_0_0_0')[-4:]
        elif self.target_rotation in ['rotFace']:
            target_quats = [rotations.euler2quat(r) for r in np.array([[0, 0, np.pi/2.],
                                                                       [0, 0, -np.pi/2.],
                                                                       [np.pi / 2., 0, 0],
                                                                       [-np.pi / 2., 0, 0],
                                                                       [0, np.pi/2., 0],
                                                                       [0, -np.pi/2., 0]])]
            parallel_quat = target_quats[self.np_random.randint(len(target_quats))]
            target_quat = self.sim.data.get_joint_qpos('rubik:free_joint_0_0_0')[-4:]
            target_quat = rotations.quat_mul(target_quat, parallel_quat)
        else:
            raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))
        assert target_quat.shape == (4,)

        target_quat /= np.linalg.norm(target_quat)
        goal = np.concatenate([target_pos, target_quat])
        return goal

def test():
    env = CubeRotEnv()
    env.reset()
    for _ in range(10):
        env.reset()
        for _ in range(10): env.render()
        env.render()
        for i in range(100):
            obs, reward, done, info = env.step(env.action_space.sample())
            env.render()
            if done:
                print("Success")
                break


if __name__ == '__main__':
    test()
