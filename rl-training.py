import os
import rlgym
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import EvalCallback
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards import BallYCoordinateReward
from rlgym.utils.obs_builders import AdvancedObs

# from observationBuilders.CustomObsBuilderBluePerspective import (
#     CustomObsBuilderBluePerspective,
# )

# set up the folders
model_name = "PPO-rl4"
models_dir = f"models/{model_name}"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# set up the RLGym environment
gym_env = rlgym.make(
    use_injector=True,
    self_play=True,
    reward_fn=VelocityBallToGoalReward(),
    obs_builder=AdvancedObs(),
    # obs_builder=CustomObsBuilderBluePerspective(),
)

# wrap the RLGym environment with the single instance wrapper
env = SB3SingleInstanceEnv(gym_env)

# create a PPO instance and start learning
learner = PPO(policy="MlpPolicy", env=env, verbose=1, tensorboard_log=logdir)

# Define a callback to log rewards to TensorBoard
eval_callback = EvalCallback(
    env,
    best_model_save_path=models_dir,
    log_path=logdir,
    eval_freq=10000,
    deterministic=True,
    render=False,  # Set to True to render the environment during evaluation
)


TIMESTEPS = 10000
iters = 0
for i in range(1, 30):
    learner.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name=model_name,
        callback=eval_callback,
    )
    learner.save(f"{models_dir}/{TIMESTEPS*i}")
