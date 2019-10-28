#!/usr/bin/env python
'''
Env for cube rotation operation
'''
import numpy as np

from gym import utils
from gym import error, spaces
from gym.envs.robotics import rotations

from manipulation.utils.util import shadow_get_obs
from manipulation.envs.manipulate_base_env import ManipulateEnv
try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class LayerOpEnv(ManipulateEnv, utils.EzPickle, ):
    def __init__(self, rot_layer, model_path='rubik_cube_shadow_wo_target.xml', target_position='ignore', target_rotation='xyz',
                 randomize_initial_position=True, randomize_initial_rotation=True):
        """
        Initializes a new ManipulateCubeEnv environment for cube rotation operation.

        Args:
            rot_layer (string): the target layer to operate
                - up: upper layer operation clockwise
                - up': upper layer operation anticlockwise
                - right: right layer operation clockwise
                - right': right layer operation anticlockwise
                - front: front layer operation clockwise
                - front': front layer operation anticlockwise
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
        """
        self.rot_layer = rot_layer
        self.goal = np.zeros(1)  # target layer angle

        assert self.rot_layer in ["up", "up'", "right","right'", "front","front'"]

        ManipulateEnv.__init__(self, target_position=target_position,
                               target_rotation=target_rotation, model_path=model_path,
                               randomize_initial_position=randomize_initial_position,
                               randomize_initial_rotation=randomize_initial_rotation,
                               )
        self.goal = self._sample_goal()

        utils.EzPickle.__init__(self)

    def _get_obs(self):
        """
        hand joint angle (24), hand joint vel (24), cube center vel (6),
        cube center position (3) + rotation (4) + layer angle (1)
        total 62
        """
        robot_qpos, robot_qvel = shadow_get_obs(self.sim, name='robot0')
        achieved_goal = self._get_achieved_goal()
        cube_vel = self.sim.data.get_joint_qvel('rubik:free_joint_0_0_0').copy()
        observation = np.concatenate([robot_qpos, robot_qvel, cube_vel, achieved_goal])
        assert observation.shape == (62,)
        angle = achieved_goal[-1:]
        return {
            'observation': observation.copy(),
            'achieved_goal': angle.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _get_achieved_goal(self):
        """
        Cube center position (3), cube center rotation (4), layer angle (1)
        total 8.
        """
        # Cube center position and rotation (quaternion).
        cube_qpos = self.sim.data.get_site_xpos('rubik:cube_center')
        cube_rot = rotations.mat2quat(self.sim.data.get_site_xmat('rubik:cube_center'))

        # Cube ball joint angles
        cube_joint_pos, cube_joint_vel = shadow_get_obs(self.sim, name='rubik')
        assert cube_joint_pos.shape == (8,)

        if self.rot_layer in ["up", "up'"]:
            box_0_0_1_euler = rotations.quat2euler(np.round(cube_joint_pos[4], 2))
            box_1_0_1_euler = rotations.quat2euler(np.round(cube_joint_pos[5], 2))
            box_0_1_1_euler = rotations.quat2euler(np.round(cube_joint_pos[6], 2))
            box_1_1_1_euler = rotations.quat2euler(np.round(cube_joint_pos[7], 2))

            for data in [box_0_0_1_euler, box_1_0_1_euler, box_0_1_1_euler, box_1_1_1_euler]:
                data[data < -3] += 6.28

            U_x_angle = [box_1_1_1_euler[0], box_1_0_1_euler[0], box_0_1_1_euler[0], box_0_0_1_euler[0]]
            U_y_angle = [box_1_1_1_euler[1], box_1_0_1_euler[1], box_0_1_1_euler[1], box_0_0_1_euler[1]]
            U_z_angle = [box_1_1_1_euler[2], box_1_0_1_euler[2], box_0_1_1_euler[2], box_0_0_1_euler[2]]

            U_bool = ((np.max(U_x_angle) - np.min(U_x_angle)) < 0.1) & \
                     ((np.max(U_y_angle) - np.min(U_y_angle)) < 0.1) & \
                     ((np.max(U_z_angle) - np.min(U_z_angle)) < 0.1)

            if U_bool:
                angle = [U_z_angle[0]]
            else:
                angle = [0]

        elif self.rot_layer in ["right", "right'"]:
            box_0_1_0_euler = rotations.quat2euler(np.round(cube_joint_pos[2], 2))
            box_1_1_0_euler = rotations.quat2euler(np.round(cube_joint_pos[3], 2))
            box_0_1_1_euler = rotations.quat2euler(np.round(cube_joint_pos[6], 2))
            box_1_1_1_euler = rotations.quat2euler(np.round(cube_joint_pos[7], 2))

            for data in [box_0_1_0_euler, box_1_1_0_euler, box_0_1_1_euler, box_1_1_1_euler]:
                data[data < -3] += 6.28

            R_x_angle = [box_1_1_1_euler[0], box_1_1_0_euler[0], box_0_1_1_euler[0], box_0_1_0_euler[0]]
            R_y_angle = [box_1_1_1_euler[1], box_1_1_0_euler[1], box_0_1_1_euler[1], box_0_1_0_euler[1]]
            R_z_angle = [box_1_1_1_euler[2], box_1_1_0_euler[2], box_0_1_1_euler[2], box_0_1_0_euler[2]]

            R_bool = ((np.max(R_x_angle) - np.min(R_x_angle)) < 0.1) & \
                     ((np.max(R_y_angle) - np.min(R_y_angle)) < 0.1) & \
                     ((np.max(R_z_angle) - np.min(R_z_angle)) < 0.1)

            if R_bool:
                angle = [R_y_angle[0]]
            else:
                angle = [0]

        elif self.rot_layer in ["front", "front'"]:
            box_1_0_0_euler = rotations.quat2euler(np.round(cube_joint_pos[1], 2))
            box_1_1_0_euler = rotations.quat2euler(np.round(cube_joint_pos[3], 2))
            box_1_0_1_euler = rotations.quat2euler(np.round(cube_joint_pos[5], 2))
            box_1_1_1_euler = rotations.quat2euler(np.round(cube_joint_pos[7], 2))

            for data in [box_1_0_0_euler, box_1_1_0_euler, box_1_0_1_euler, box_1_1_1_euler]:
                data[data < -3] += 6.28

            F_x_angle = [box_1_0_1_euler[0], box_1_0_0_euler[0], box_1_1_1_euler[0], box_1_1_0_euler[0]]
            F_y_angle = [box_1_0_1_euler[1], box_1_0_0_euler[1], box_1_1_1_euler[1], box_1_1_0_euler[1]]
            F_z_angle = [box_1_0_1_euler[2], box_1_0_0_euler[2], box_1_1_1_euler[2], box_1_1_0_euler[2]]

            F_bool = ((np.max(F_x_angle) - np.min(F_x_angle)) < 0.1) & \
                     ((np.max(F_y_angle) - np.min(F_y_angle)) < 0.1) & \
                     ((np.max(F_z_angle) - np.min(F_z_angle)) < 0.1)

            if F_bool:
                angle = [F_x_angle[0]]
            else:
                angle = [0]

        cube_qpos = np.concatenate([cube_qpos, cube_rot, angle])
        assert cube_qpos.shape == (8,)
        return cube_qpos

    def _goal_distance(self, goal_a, goal_b):
        """
        return: angle difference absolute value.
        """
        assert goal_a.shape[-1] == 1
        assert goal_b.shape[-1] == 1

        delta_angle = np.abs(goal_a[..., 0] - goal_b[..., 0])

        return delta_angle

    def _is_success(self, achieved_goal, desired_goal):
        d_angle = self._goal_distance(achieved_goal, desired_goal)
        achieved_angle = (d_angle < self.rotation_threshold).astype(np.float32)
        cube_center_pos = self.sim.data.get_site_xpos('rubik:cube_center')
        achieved_height = (cube_center_pos[2] > 0.1).astype(np.float32)
        success = achieved_angle * achieved_height
        return success

    def _sample_goal(self):
        """Uniformly sample an angle (in radian) in [-1.57, 1.57]"""
        if self.rot_layer.endswith("'"):
            goal = self.np_random.uniform(-np.pi/2, 0, 1)
        else:
            goal = self.np_random.uniform(0, np.pi/2, 1)
        return goal


class LayerOpEnvUp(LayerOpEnv):
    def __init__(self):
        super(LayerOpEnvUp, self).__init__(
            rot_layer='up',
            target_position='ignore',
            target_rotation='ignore',
            randomize_initial_position=True,
            randomize_initial_rotation=False)

class LayerOpEnvUpPrime(LayerOpEnv):
    def __init__(self):
        super(LayerOpEnvUpPrime, self).__init__(
            rot_layer="up'",
            target_position='ignore',
            target_rotation='ignore',
            randomize_initial_position=True,
            randomize_initial_rotation=False)

class LayerOpEnvRight(LayerOpEnv):
    def __init__(self):
        super(LayerOpEnvRight, self).__init__(
            rot_layer='right',
            target_position='ignore',
            target_rotation='ignore',
            randomize_initial_position=True,
            randomize_initial_rotation=False)

class LayerOpEnvRightPrime(LayerOpEnv):
    def __init__(self):
        super(LayerOpEnvRightPrime, self).__init__(
            rot_layer="right'",
            target_position='ignore',
            target_rotation='ignore',
            randomize_initial_position=True,
            randomize_initial_rotation=False)

class LayerOpEnvFront(LayerOpEnv):
    def __init__(self):
        super(LayerOpEnvFront, self).__init__(
            rot_layer='front',
            target_position='ignore',
            target_rotation='ignore',
            randomize_initial_position=True,
            randomize_initial_rotation=False)

class LayerOpEnvFrontPrime(LayerOpEnv):
    def __init__(self):
        super(LayerOpEnvFrontPrime, self).__init__(
            rot_layer="front'",
            target_position='ignore',
            target_rotation='ignore',
            randomize_initial_position=True,
            randomize_initial_rotation=False)

def test():
    env = LayerOpEnvFrontPrime()
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
