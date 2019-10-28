#!/usr/bin/env python
"""
A demo for cube solver
"""
import mujoco_py
import numpy as np
from os import path
from manipulation.utils.solver import solveCube

class CubeSolver:
    def __init__(self, model_path, state_path):
        assert path.exists(model_path), "Model {} does not exist!".format(model_path)
        assert path.exists(state_path), "Scrambled states {} does not exist!".format(state_path)
        self.model = mujoco_py.load_model_from_path(model_path)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = mujoco_py.MjViewer(self.sim)
        self.states = np.load(state_path, allow_pickle=True)

    def reset(self):
        '''Randomly pick one scrambled pattern'''
        scrambed_dist = np.random.randint(len(self.states))
        scrambed_pattern = np.random.randint(len(self.states[scrambed_dist]))
        pos_dict = self.states[scrambed_dist][scrambed_pattern]
        for name, value in pos_dict.items():
            self.data.set_joint_qpos(name, value)
        self.sim.forward()
        self.rotate_cube('reset')

    def solve(self):
        '''Solve the scrambled cube, return planned trajectory'''
        s = self._get_cube_state()
        moves = solveCube(s)
        return moves

    def rotate_cube(self, cmd):
        assert cmd in ["U", "U'", "U2", "R", "R'", "R2", "F", "F'", "F2", 'reset', 'show']
        t_1 = 0.1
        t_2 = 0.4
        x_in = -0.03
        x_out = 0.02

        current_q = np.array(self.sim.data.actuator_length)
        goal_q = current_q.copy()
        if cmd == 'reset':
            goal_q[4:8] = x_out
            goal_q[12:16] = x_out
            goal_q[20:24] = x_out
            current_q = self._go_to_q_pos(goal_q, current_q, t_1)

        if cmd == 'show':
            current_q = self._go_to_q_pos(goal_q, current_q, 10)

        elif cmd == 'U':
            goal_q[4:8] = x_in
            current_q = self._go_to_q_pos(goal_q, current_q, t_1)
            goal_q = current_q.copy()
            goal_q[:4] += np.pi / 2.0
            current_q = self._go_to_q_pos(goal_q, current_q, t_2)
        elif cmd == "U'":
            goal_q[4:8] = x_in
            current_q = self._go_to_q_pos(goal_q, current_q, t_1)
            goal_q = current_q.copy()
            goal_q[:4] -= np.pi / 2.0
            current_q = self._go_to_q_pos(goal_q, current_q, t_2)
        elif cmd == "U2":
            goal_q[4:8] = x_in
            current_q = self._go_to_q_pos(goal_q, current_q, t_1)
            goal_q = current_q.copy()
            goal_q[:4] += np.pi
            current_q = self._go_to_q_pos(goal_q, current_q, t_2 * 2)

        elif cmd == 'R':
            goal_q[12:16] = x_in
            current_q = self._go_to_q_pos(goal_q, current_q, t_1)
            goal_q = current_q.copy()
            goal_q[8:12] += np.pi / 2.0
            current_q = self._go_to_q_pos(goal_q, current_q, t_2)
        elif cmd == "R'":
            goal_q[12:16] = x_in
            current_q = self._go_to_q_pos(goal_q, current_q, t_1)
            goal_q = current_q.copy()
            goal_q[8:12] -= np.pi / 2.0
            current_q = self._go_to_q_pos(goal_q, current_q, t_2)
        elif cmd == "R2":
            goal_q[12:16] = x_in
            current_q = self._go_to_q_pos(goal_q, current_q, t_1)
            goal_q = current_q.copy()
            goal_q[8:12] += np.pi
            current_q = self._go_to_q_pos(goal_q, current_q, t_2 * 2)

        elif cmd == 'F':
            goal_q[20:24] = x_in
            current_q = self._go_to_q_pos(goal_q, current_q, t_1)
            goal_q = current_q.copy()
            goal_q[16:20] += np.pi / 2.0
            current_q = self._go_to_q_pos(goal_q, current_q, t_2)
        elif cmd == "F'":
            goal_q[20:24] = x_in
            current_q = self._go_to_q_pos(goal_q, current_q, t_1)
            goal_q = current_q.copy()
            goal_q[16:20] -= np.pi / 2.0
            current_q = self._go_to_q_pos(goal_q, current_q, t_2)
        elif cmd == "F2":
            goal_q[20:24] = x_in
            current_q = self._go_to_q_pos(goal_q, current_q, t_1)
            goal_q = current_q.copy()
            goal_q[16:20] += np.pi
            current_q = self._go_to_q_pos(goal_q, current_q, t_2 * 2)

        goal_q = current_q.copy()
        goal_q[4:8] = x_out
        goal_q[12:16] = x_out
        goal_q[20:24] = x_out
        self._go_to_q_pos(goal_q, current_q, 0.2)

    def _go_to_q_pos(self, goal_q, current_q, time, time_step=0.0002):
        """Send the command to rotate the Rubik's cube"""
        step_num = time / time_step
        for i in range(int(np.floor(step_num)) - 1):
            self.data.ctrl[:] = current_q + (i + 1) * (goal_q - current_q) / step_num
            self.sim.step()
            if i % 10 == 0:     # for faster visualization
                self.viewer.render()

        self.data.ctrl[:] = goal_q
        self.sim.step()
        self.viewer.render()
        return goal_q.copy()

    def _get_face_idx(self, local_dist):
        face_dist_list = np.array([[-3, -3, 1.5],
                                   [-3, 0, 1.5],
                                   [0, -3, 1.5],
                                   [0, 0, 1.5],
                                   [0, 1.5, 0],
                                   [-3, 1.5, 0],  # 5
                                   [0, 1.5, -3],
                                   [-3, 1.5, -3],
                                   [1.5, -3, 0],
                                   [1.5, 0, 0],
                                   [1.5, -3, -3],  # 10
                                   [1.5, 0, -3],
                                   [0, -3, -4.5],
                                   [0, 0, -4.5],
                                   [-3, -3, -4.5],
                                   [-3, 0, -4.5],  # 15
                                   [-3, -4.5, 0],
                                   [0, -4.5, 0],
                                   [-3, -4.5, -3],
                                   [0, -4.5, -3],
                                   [-4.5, 0, 0],  # 20
                                   [-4.5, -3, 0],
                                   [-4.5, 0, -3],
                                   [-4.5, -3, -3]
                                   ]) + np.array([[3, 3, 3]])
        dist_list = np.square(local_dist[0] - face_dist_list[:, 0]) + \
                    np.square(local_dist[1] - face_dist_list[:, 1]) + \
                    np.square(local_dist[2] - face_dist_list[:, 2])
        return np.argmin(dist_list)

    def _get_cube_state(self):
        '''
        0: white, 1: red, 2: green, 3: yellow, 4: orange, 5: blue
        '''
        s = np.zeros(24, dtype=np.int8)
        color_idx = {'white': 0, 'red': 1, 'green': 2, 'yellow': 3, 'orange': 4, 'blue': 5}
        patches = {'rubik:box_1_1_1_w': 'white', 'rubik:box_1_1_1_g': 'green', 'rubik:box_1_1_1_r': 'red',
                   'rubik:box_1_0_1_w': 'white', 'rubik:box_1_0_1_g': 'green', 'rubik:box_1_0_1_o': 'orange',
                   'rubik:box_0_1_1_w': 'white', 'rubik:box_0_1_1_b': 'blue', 'rubik:box_0_1_1_r': 'red',
                   'rubik:box_0_0_1_w': 'white', 'rubik:box_0_0_1_b': 'blue', 'rubik:box_0_0_1_o': 'orange',
                   'rubik:box_1_1_0_y': 'yellow', 'rubik:box_1_1_0_g': 'green', 'rubik:box_1_1_0_r': 'red',
                   'rubik:box_1_0_0_y': 'yellow', 'rubik:box_1_0_0_g': 'green', 'rubik:box_1_0_0_o': 'orange',
                   'rubik:box_0_1_0_y': 'yellow', 'rubik:box_0_1_0_b': 'blue', 'rubik:box_0_1_0_r': 'red',
                   'rubik:box_0_0_0_y': 'yellow', 'rubik:box_0_0_0_b': 'blue', 'rubik:box_0_0_0_o': 'orange'}
        box1_rot_mat = self.data.get_body_xmat('rubik:box_0_0_0').T
        box1_pos = self.data.get_body_xpos('rubik:box_0_0_0')

        # Step 1. fixed box_1_1_1
        s[9] = color_idx['green']
        s[4] = color_idx['red']
        s[3] = color_idx['white']

        # Step 2. box_1_0_1
        for patch_name in patches.keys():
            patch_pos = self.data.get_geom_xpos(patch_name)
            local_pos = np.matmul(box1_rot_mat, patch_pos - box1_pos)
            idx = self._get_face_idx(np.round(local_pos * 100, 3))
            s[idx] = color_idx[patches[patch_name]]
        return s

def test():
    model_path = path.join(path.dirname(__file__), "../envs/assets/rubik_cube.xml")     # use rubik_cube_bar.xml if you want to know what happened
    state_path = path.join(path.dirname(__file__), "../data/scrambled_cubes.npy")
    solver = CubeSolver(model_path, state_path)
    for i in range(5):
        print("Solving the cube {}/{}".format(i+1, 10))
        solver.reset()
        moves = solver.solve()
        for move in moves:
            solver.rotate_cube(move)


if __name__ == '__main__':
    test()




