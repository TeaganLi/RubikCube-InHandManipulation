#!/usr/bin/env python
"""
A demo for cube rotation and layer-wise operation tasks with trained policy
"""
from os import path
import pickle, argparse
from manipulation.envs.manipulate_cube_env import CubeRotEnv
from manipulation.envs.manipulate_layer_env import LayerOpEnvUpPrime

def test():
    parser = argparse.ArgumentParser(description="Demo for cube rotation and layer-wise operation tasks.")
    parser.add_argument("--env", type=str, default='CubeRot',
                        help="CubeRot or LayerOp")
    parser.add_argument("--num_epi", type=int, default=10,
                        help="number of the demo episodes")
    args = parser.parse_args()
    assert args.env in ['CubeRot', 'LayerOp'], "env should be CubeRot or LayerOp"
    if args.env == 'CubeRot': env = CubeRotEnv()
    else: env = LayerOpEnvUpPrime()
    policy_path = path.join(path.dirname(__file__), "../data", args.env) + ".pkl"
    with open(policy_path, 'rb') as f:
        policy = pickle.load(f)

    for j in range(args.num_epi):
        env.reset()
        for _ in range(10):
            env.render()
        env.render()
        print("Demo epi: {}/{}".format(j+1, args.num_epi))
        for i in range(100):
            obs = env._get_obs()
            o, ag, g = obs['observation'], obs['achieved_goal'], obs['desired_goal']
            output = policy.get_actions(o, ag, g)
            obs, reward, done, info = env.step(output)
            env.render()
            if info['is_success']:
                break


if __name__ == '__main__':
    test()

