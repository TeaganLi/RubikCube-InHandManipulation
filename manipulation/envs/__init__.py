from gym.envs.registration import registry, register, make, spec

register(
         id='LayerOpUpEnv-v0',
         entry_point='manipulation.envs.manipulate_layer_env:LayerOpEnvUp',
         max_episode_steps=100,
         )

register(
         id='LayerOpUpPrimeEnv-v0',
         entry_point='manipulation.envs.manipulate_layer_env:LayerOpEnvUpPrime',
         max_episode_steps=100,
         )

register(
         id='LayerOpRightEnv-v0',
         entry_point='manipulation.envs.manipulate_layer_env:LayerOpRight',
         max_episode_steps=100,
         )

register(
         id='LayerOpRightPrimeEnv-v0',
         entry_point='manipulation.envs.manipulate_layer_env:LayerOpRightPrime',
         max_episode_steps=100,
         )

register(
         id='LayerOpFrontEnv-v0',
         entry_point='manipulation.envs.manipulate_layer_env:LayerOpFront',
         max_episode_steps=100,
         )

register(
         id='LayerOpFrontPrimeEnv-v0',
         entry_point='manipulation.envs.manipulate_layer_env:LayerOpFrontPrime',
         max_episode_steps=100,
         )

register(
         id='CubeRotEnv-v0',
         entry_point='manipulation.envs.manipulate_cube_env:CubeRotEnv',
         max_episode_steps=100,
         )

