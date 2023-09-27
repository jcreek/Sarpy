import os
import rlgym
from stable_baselines3.ppo import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import TensorBoardOutputFormat, configure
from torch.utils.tensorboard import SummaryWriter
from rlgym_tools.sb3_utils import SB3SingleInstanceEnv
from rlgym.utils.reward_functions.common_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards import BallYCoordinateReward
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.terminal_conditions.common_conditions import (
    TimeoutCondition,
    GoalScoredCondition,
)

# from observationBuilders.CustomObsBuilderBluePerspective import (
#     CustomObsBuilderBluePerspective,
# )

# Set up the folders
model_name = "PPO-rl6"
models_dir = f"models/{model_name}"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# Set up the RLGym environment
gym_env = rlgym.make(
    use_injector=True,
    self_play=True,
    reward_fn=BallYCoordinateReward(),
    obs_builder=AdvancedObs(),
    terminal_conditions=(TimeoutCondition(225), GoalScoredCondition()),
)

# Wrap the RLGym environment with the single instance wrapper
env = SB3SingleInstanceEnv(gym_env)

# Create a PPO instance
learner = PPO(policy="MlpPolicy", env=env, verbose=1, tensorboard_log=logdir)

# Configure TensorBoard for custom logging
tb_log_name = f"logs/{model_name}"
tensorboard_writer = SummaryWriter(log_dir=tb_log_name)

# Define a callback to log rewards to TensorBoard
eval_callback = EvalCallback(
    env,
    best_model_save_path=models_dir,
    log_path=logdir,
    eval_freq=10000,
    deterministic=True,
    render=False,  # Set to True to render the environment during evaluation
)


# Custom callback for saving the model every 100 episodes
def custom_callback(locals, globals):
    if num_episodes % 100 == 0:
        model_save_path = f"{models_dir}/episode_{num_episodes}"
        locals["self"].save(model_save_path)


# Training loop
EPISODES = 1000  # Set the total number of episodes
num_episodes = 0

while num_episodes < EPISODES:
    obs = env.reset()
    episode_reward = 0
    episode_length = 0

    while True:
        action, _ = learner.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += sum(reward)
        episode_length += 1

        print(done)

        if done.any():  # Check if any element in 'done' is True
            num_episodes += 1  # Increment the episode count
            break

    # Calculate the average reward per episode
    avg_episode_reward = episode_reward / episode_length

    # Log custom episode-related information to TensorBoard
    print(episode_reward)
    tensorboard_writer.add_scalar("Episode/Reward", float(episode_reward), num_episodes)
    tensorboard_writer.add_scalar("Episode/Length", episode_length, num_episodes)

    # Run the evaluation callback
    if num_episodes % 50 == 0:
        eval_callback.on_epoch_end()

    # Use the custom callback for saving the model
    custom_callback(locals(), globals())
