import os
import numpy as np
import atexit

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan

from rlgym.envs import Match

from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.reward_functions.common_rewards import (
    VelocityBallToGoalReward,
    LiuDistanceBallToGoalReward,
    BallYCoordinateReward,
    EventReward,
    RewardIfClosestToBall,
    LiuDistancePlayerToBallReward,
    FaceBallReward,
    TouchBallReward,
    AlignBallGoal,
)
from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import (
    TimeoutCondition,
    NoTouchTimeoutCondition,
    GoalScoredCondition,
)
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import (
    VelocityPlayerToBallReward,
)
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import (
    VelocityBallToGoalReward,
)
from rlgym.utils.reward_functions import CombinedReward

from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder

if __name__ == "__main__":  # Required for multiprocessing
    # Set up the folders
    models_dir = "models"
    log_dir = "logs"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    frame_skip = 8  # Number of ticks to repeat an action
    half_life_seconds = (
        5  # Easier to conceptualize, after this many seconds the reward discount is 0.5
    )

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    print(f"fps={fps}, gamma={gamma})")

    def exit_save(model):
        model.save("models/exit_save")

    team_size = 3

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=team_size,
            tick_skip=frame_skip,
            reward_function=CombinedReward(
                (
                    VelocityPlayerToBallReward(),
                    VelocityBallToGoalReward(),
                    LiuDistanceBallToGoalReward(),
                    EventReward(
                        team_goal=100.0,
                        concede=-100.0,
                        shot=5.0,
                        save=30.0,
                        demo=10.0,
                    ),
                    BallYCoordinateReward(),
                    RewardIfClosestToBall(LiuDistancePlayerToBallReward()),
                    LiuDistancePlayerToBallReward(),
                    FaceBallReward(),
                    TouchBallReward(),
                    AlignBallGoal(),
                    KickoffReward(),
                ),
                (1.0, 1.0, 50.0, 1.0, 0.1, 0.2, 0.1, 0.2, 10.0, 0.1, 10.0),
            ),
            spawn_opponents=True,
            terminal_conditions=[
                TimeoutCondition(fps * 300),
                NoTouchTimeoutCondition(fps * 10),
                GoalScoredCondition(),
            ],
            obs_builder=AdvancedObsPadder(
                team_size=team_size
            ),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction(),  # Discrete > Continuous don't @ me
        )

    # Generate the environment (the Rocket League game used by RL Gym)
    env = SB3MultipleInstanceEnv(
        get_match, 10
    )  # Start 10 instances, waiting 60 seconds between each
    env = VecCheckNan(env)  # Optional
    env = VecMonitor(env)  # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(
        env, norm_obs=False, gamma=gamma
    )  # Highly recommended, normalizes rewards

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(
        round(1_000_000 / env.num_envs), save_path=models_dir, name_prefix="sarpy_model"
    )

    try:
        while True:
            # Load the model that was last trained, or start a new one if the zip file doesn't exist
            try:
                # Now, if one wants to load a trained model from a checkpoint, use this function
                # This will contain all the attributes of the original model
                # Any attribute can be overwritten by using the custom_objects parameter,
                # which includes n_envs (number of agents), which has to be overwritten to use a different amount
                model = PPO.load(
                    f"{models_dir}/exit_save.zip",
                    env,
                    custom_objects=dict(
                        n_envs=env.num_envs, _last_obs=None
                    ),  # Need this to change number of agents
                    device="auto",  # Need to set device again (if using a specific one)
                    force_reset=True,  # Make SB3 reset the env so it doesn't think we're continuing from last state
                )
                print("Loaded exit_save.zip model")
            except:
                model = PPO(
                    MlpPolicy,
                    env,
                    n_epochs=32,  # PPO calls for multiple epochs
                    learning_rate=1e-5,  # Around this is fairly common for PPO
                    ent_coef=0.01,  # From PPO Atari
                    vf_coef=1.0,  # From PPO Atari
                    gamma=gamma,  # Gamma as calculated using half-life
                    verbose=3,  # Print out all the info as we're going
                    batch_size=4096,  # Batch size as high as possible within reason
                    n_steps=4096,  # Number of steps to perform before optimizing network
                    tensorboard_log="out/logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
                    device="auto",  # Uses GPU if available
                )
                print("Created new model")

            atexit.register(exit_save, model)

            print("Learning...")
            model.learn(100_000_000, callback=callback, reset_num_timesteps=False)
            model.save(f"{models_dir}/exit_save")
            print("Saved exit_save model")

    except Exception as e:
        print(e)
