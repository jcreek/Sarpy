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
    BallYCoordinateReward,
    EventReward,
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


# from observationBuilders.CustomObsBuilderBluePerspective import (
#     CustomObsBuilderBluePerspective,
# )

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
    agents_per_match = 2
    num_instances = 10
    target_steps = 100_000
    steps = target_steps // (num_instances * agents_per_match)
    batch_size = steps

    def exit_save(model):
        model.save("models/exit_save")

    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=1,
            tick_skip=frame_skip,
            reward_function=CombinedReward(
                (
                    VelocityPlayerToBallReward(),
                    VelocityBallToGoalReward(),
                    EventReward(
                        team_goal=100.0,
                        concede=-100.0,
                        shot=5.0,
                        save=30.0,
                        demo=10.0,
                    ),
                    BallYCoordinateReward(),
                ),
                (0.1, 1.0, 1.0, 0.1),
            ),
            # self_play=True,
            terminal_conditions=[
                TimeoutCondition(fps * 300),
                NoTouchTimeoutCondition(fps * 20),
                GoalScoredCondition(),
            ],
            obs_builder=AdvancedObs(),  # Not that advanced, good default
            state_setter=DefaultState(),  # Resets to kickoff position
            action_parser=DiscreteAction(),  # Discrete > Continuous don't @ me
        )

    # Generate the environment (the Rocket League game used by RL Gym)
    env = SB3MultipleInstanceEnv(
        get_match, num_instances
    )  # Start x instances, waiting 60 seconds between each
    env = VecCheckNan(env)  # Optional
    env = VecMonitor(env)  # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(
        env, norm_obs=False, gamma=gamma
    )  # Highly recommended, normalizes rewards

    # Load the model that was last trained, or start a new one if the zip file doesn't exist
    try:
        model = PPO.load(
            f"{models_dir}/exit_save.zip",
            env,
            device="auto",  # Need to set device again (if using a specific one)
        )
        print("Loaded exit_save.zip model")
    except:
        model = PPO(
            MlpPolicy,
            env,
            learning_rate=5e-5,  # Around this is fairly common for PPO
            n_steps=steps,  # Number of steps to perform before optimizing network
            batch_size=batch_size,  # Batch size as high as possible within reason
            n_epochs=1,  # PPO calls for multiple epochs
            gamma=gamma,  # Gamma as calculated using half-life
            ent_coef=0.01,  # From PPO Atari
            vf_coef=1.0,  # From PPO Atari
            tensorboard_log=log_dir,  # `tensorboard --logdir out/logs` in terminal to see graphs
            verbose=3,  # Print out all the info as we're going
            device="auto",  # Uses GPU if available
        )
        print("Created new model")

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(
        round(5_000_000 / env.num_envs), save_path=models_dir, name_prefix="rl_model"
    )

    atexit.register(exit_save, model)

    try:
        while True:
            print("Learning...")
            model.learn(25_000_000, callback=callback)
            model.save(f"{models_dir}/exit_save")
            print("Saved exit_save model")
            model.save(f"mmr_models/{model.num_timesteps}")
    except Exception as e:
        print(e)
