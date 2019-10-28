#!/usr/bin/env python
'''
Base env for shadow hand manipulator
'''
import numpy as np

from gym import utils
from gym import error, spaces
from gym.envs.robotics import rotations

from manipulation.envs.base_env import ShadowGoalBaseEnv, ActionRangeType
from manipulation.utils.util import quat_from_angle_and_axis

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(e))


DEFAULT_INITIAL_QPOS = {
    'robot0:WRJ1': -0.1745,
    'robot0:WRJ0': -0.1045,
    'robot0:FFJ3': 0.,
    'robot0:FFJ2': 0.7855,
    'robot0:FFJ1': 0.7855,
    'robot0:MFJ3': 0,
    'robot0:MFJ2': 0.7855,
    'robot0:MFJ1': 0.7855,
    'robot0:RFJ3': 0,
    'robot0:RFJ2': 0.7855,
    'robot0:RFJ1': 0.7855,
    'robot0:LFJ4': 0.3925,
    'robot0:LFJ3': 0,
    'robot0:LFJ2': 0.7855,
    'robot0:LFJ1': 0.7855,
    'robot0:THJ4': 0,
    'robot0:THJ3': 0.611,
    'robot0:THJ2': -0.,
    'robot0:THJ1': -0.,
    'robot0:THJ0': -0.7855,
}


class ManipulateEnv(ShadowGoalBaseEnv, utils.EzPickle):
    def __init__(self, target_position, target_rotation, skip_frame=20,
                 model_path='rubik_cube_shadow_wo_target.xml', initial_robot_pos_dict=DEFAULT_INITIAL_QPOS,
                 reward_type='sparse', randomize_initial_position=True, randomize_initial_rotation=True,
                 distance_threshold=0.01, rotation_threshold=0.1,
                 target_position_range=np.array([(-0.04, 0.04), (-0.06, 0.02), (0.0, 0.06)])):
        """Initializes a new ManipulateEnv environment.

        Args:
            model_path (string): path to the environment XML file
            initial_robot_pos_dict (dict): a dictionary of joint names and values that define the initial configuration
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
            target_position_range (np.array of shape (3,2)): range of the target_position randomization
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            randomize_initial_position (boolean): whether or not to randomize the initial position of the object
            randomize_initial_rotation (boolean): whether or not to randomize the initial rotation of the object
            distance_threshold (float, in meters): the threshold after which the position of a goal is considered achieved
            rotation_threshold (float, in radians): the threshold after which the rotation of a goal is considered achieved
        """
        self.target_position = target_position
        self.target_rotation = target_rotation
        self.target_position_range = target_position_range
        self.reward_type = reward_type
        self.randomize_initial_position = randomize_initial_position
        self.randomize_initial_rotation = randomize_initial_rotation
        self.distance_threshold = distance_threshold
        self.rotation_threshold = rotation_threshold
        self.parallel_quats = [rotations.euler2quat(r) for r in rotations.get_parallel_rotations()]

        assert self.reward_type in ['dense', 'sparse']
        assert self.target_position in ['ignore', 'fixed', 'random']
        assert self.target_rotation in ['ignore', 'fixed', 'xyz', 'z', 'parallel', 'rotFace']

        ShadowGoalBaseEnv.__init__(self, model_path, initial_robot_pos_dict, skip_frame)

    def _get_action_space(self):
        return spaces.Box(-1.0, 1.0, shape=(20,), dtype='float32')

    def _get_observation_space(self):
        obs = self._get_obs()
        observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['desired_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        return observation_space

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.sim.data.ctrl[:] = self.get_current_actuator_pos()
        self.sim.forward()

        initial_qpos = self.sim.data.get_joint_qpos('rubik:free_joint_0_0_0').copy()
        initial_pos, initial_quat = initial_qpos[:3], initial_qpos[3:]
        assert initial_qpos.shape == (7,)
        assert initial_pos.shape == (3,)
        assert initial_quat.shape == (4,)

        # Randomize initial rotation
        if self.randomize_initial_rotation:
            if self.target_rotation == 'z':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'parallel':
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = np.array([0., 0., 1.])
                z_quat = quat_from_angle_and_axis(angle, axis)
                parallel_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
                offset_quat = rotations.quat_mul(z_quat, parallel_quat)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation in ['xyz', 'ignore']:
                angle = self.np_random.uniform(-np.pi, np.pi)
                axis = self.np_random.uniform(-1., 1., size=3)
                offset_quat = quat_from_angle_and_axis(angle, axis)
                initial_quat = rotations.quat_mul(initial_quat, offset_quat)
            elif self.target_rotation == 'fixed':
                pass
            elif self.target_rotation == 'rotFace':
                initial_quat = self.parallel_quats[self.np_random.randint(len(self.parallel_quats))]
            else:
                raise error.Error('Unknown target_rotation option "{}".'.format(self.target_rotation))

        # Randomize initial position
        if self.randomize_initial_position:
            delta = self.np_random.normal(size=2, scale=0.002)
            initial_pos[:2] += delta

        initial_quat /= np.linalg.norm(initial_quat)
        initial_qpos = np.concatenate([initial_pos, initial_quat])
        self.sim.data.set_joint_qpos('rubik:free_joint_0_0_0', initial_qpos)

        for _ in range(10):
            try:
                self.sim.step()
            except mujoco_py.MujocoException:
                return False
        self.goal = self._sample_goal()

        def is_on_palm():
            self.sim.forward()
            cube_center_pos = self.sim.data.get_site_xpos('rubik:cube_center')
            is_on_palm = (cube_center_pos[2] > 0.10)
            return is_on_palm

        return is_on_palm()

    def step(self, action):
        # 1. conduct a simulation step
        assert self.action_space.contains(action)
        self.take_action(action, action_range_type=ActionRangeType.Normalized)
        # 2. get obs
        obs = self._get_obs()
        # 3. is done
        done = False
        # 4. info
        info = {'is_success': self._is_success(obs['achieved_goal'], self.goal)}
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def compute_reward(self, achieved_goal, goal, info):
        if self.reward_type == 'sparse':
            success = self._is_success(achieved_goal, goal)
            return (success - 1.0)
        else:
            dist = self._goal_distance(achieved_goal, goal)
            return np.exp(-20 * dist)

    def _render_callback(self):
        # Assign current state to target object but offset a bit so that the actual object
        # is not obscured.
        goal = self.goal.copy()
        if goal.shape == (7,):
            if self.target_position == 'ignore':
                # Move the object to the side since we do not care about it's position.
                goal[0] += 0.15
            self.sim.data.set_joint_qpos('target_free_joint', goal)
            self.sim.data.set_joint_qvel('target_free_joint', np.zeros(6))

        if 'object_hidden' in self.sim.model.geom_names:
            hidden_id = self.sim.model.geom_name2id('object_hidden')
            self.sim.model.geom_rgba[hidden_id, 3] = 1.
        self.sim.forward()

    def _get_obs(self):
        raise NotImplementedError

    def _get_achieved_goal(self):
        raise NotImplementedError

    def _goal_distance(self, goal_a, goal_b):
        raise NotImplementedError

    def _is_success(self, achieved_goal, desired_goal):
        raise NotImplementedError

    def _sample_goal(self):
        raise NotImplementedError

