import rlgym
from stable_baselines3.ppo import PPO
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv
import os

# set up the folders
models_dir = "models/PPO-rl"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# set up the RLGym environment
gym_env = rlgym.make(use_injector=True, self_play=True)

# wrap the RLGym environment with the single instance wrapper
env = SB3SingleInstanceEnv(gym_env)

# create a PPO instance and start learning
learner = PPO(policy="MlpPolicy", env=env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
for i in range(1, 30):
    learner.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO-rl")
    learner.save(f"{models_dir}/{TIMESTEPS*i}")