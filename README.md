# Learning to Solve a Rubik's Cube with a Dexterous Hand

This is the environment implemenation based on mujoco of our paper: [**Learning to Solve a Rubik's Cube with a Dexterous Hand**](https://arxiv.org/abs/1907.11388).

In our paper, the whole task is decomposed into two subtasks, namely cube rotation and layer-wise operation. We provide the two subtasks here including the trained policy. You can find the [**video**](https://www.youtube.com/watch?v=4st_54rqJB0&t=8s) and [**project page**](https://sites.google.com/view/learning-solve-a-rubiks-cube/home).

If you find our project helpful, your citations are highly appreciated:

```
@article{li2019learning,
  title={Learning to Solve a Rubik's Cube with a Dexterous Hand},
  author={Li, Tingguang and Xi, Weitao and Fang, Meng and Xu, Jia and Meng, Max Qing-Hu},
  journal={arXiv preprint arXiv:1907.11388},
  year={2019}
}
```

## Prerequisites
The following platforms are currently supported:
* Linux with Python3.6+
* OS X with Python3.6+

Please make sure the following packages have been installed on your computer:
* mujoco(https://github.com/openai/mujoco-py) (license required)
* gym(https://github.com/openai/gym)
* tensorflow (if you want to load our trained policies)

## Installation
You can install the project with
```
git clone https://github.com/TeaganLi/RubikCube-InHandManipulation.git
cd RubikCube-InHandManipulation
pip install -e .
```

## Run Demos

After you installed our package and tensorflow, you can run our demos with our trained policy. 
```
cd manipulation/demo
python subtask_demo.py --env CubeRot	# run the cube rotation demo
python subtask_demo.py --env LayerOp	# run the layer-wise operation demo	
```
You can run the cube solver demo with
```
python cube_solver_demo.py
```

## Train your model
We provide the following environments based on the mujoco simulator. All of them are gym-compatible and you can easily train your model using RL libraries like baselines(https://github.com/openai/baselines).
* CubeRotEnv-v0: cube rotation environment
* LayerOpUpEnv-v0: layer-wise operation with upper layer clockwise goal
* LayerOpUpPrimeEnv-v0: layer-wise operation with upper layer anticlockwise goal
* LayerOpRightEnv-v0: layer-wise operation with right layer clockwise goal
* LayerOpRightPrimeEnv-v0: layer-wise operation with right layer anticlockwise goal
* LayerOpFrontEnv-v0: layer-wise operation with front layer clockwise goal
* LayerOpFrontPrimeEnv-v0: layer-wise operation with front layer anticlockwise goal

## License:
This project is for uncommercial use. 
Contact [**Tingguang Li**](http://www.ee.cuhk.edu.hk/~tgli/) if you have any questions.

