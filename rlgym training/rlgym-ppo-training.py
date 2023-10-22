import numpy as np
from rlgym_sim.utils.gamestates import GameState
from rlgym_ppo.util import MetricsLogger


class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [
            game_state.players[0].car_data.linear_velocity,
            game_state.players[0].car_data.rotation_mtx(),
            game_state.orange_score,
        ]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_linvel = np.zeros(3)
        for metric_array in collected_metrics:
            p0_linear_velocity = metric_array[0]
            avg_linvel += p0_linear_velocity
        avg_linvel /= len(collected_metrics)
        report = {
            "x_vel": avg_linvel[0],
            "y_vel": avg_linvel[1],
            "z_vel": avg_linvel[2],
            "Cumulative Timesteps": cumulative_timesteps,
        }
        wandb_run.log(report)


def build_rocketsim_env():
    import rlgym_sim
    from rlgym_sim.utils.reward_functions import CombinedReward
    from rlgym_sim.utils.reward_functions.common_rewards import (
        VelocityPlayerToBallReward,
        VelocityBallToGoalReward,
        EventReward,
        SaveBoostReward,
    )
    from rlgym_tools.extra_rewards.kickoff_reward import KickoffReward
    from rlgym_sim.utils.obs_builders import DefaultObs
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        NoTouchTimeoutCondition,
        GoalScoredCondition,
    )
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.action_parsers import ContinuousAction
    from rlgym_tools.extra_obs.advanced_padder import AdvancedObsPadder

    spawn_opponents = True
    team_size = 1
    game_tick_rate = 120
    tick_skip = 8
    timeout_seconds = 10
    timeout_ticks = int(round(timeout_seconds * game_tick_rate / tick_skip))

    action_parser = ContinuousAction()
    terminal_conditions = [
        NoTouchTimeoutCondition(timeout_ticks),
        GoalScoredCondition(),
    ]

    rewards_to_combine = (
        VelocityPlayerToBallReward(),
        VelocityBallToGoalReward(),
        EventReward(
            team_goal=150.0,
            concede=-100.0,
            shot=10.0,
            save=60.0,
            demo=20.0,
        ),
        KickoffReward(),
        SaveBoostReward(),
    )
    reward_weights = (1.0, 1.0, 1.0, 1.0, 1.0)

    reward_fn = CombinedReward(
        reward_functions=rewards_to_combine, reward_weights=reward_weights
    )

    obs_builder = AdvancedObsPadder(team_size=team_size)

    # DefaultObs(
    #     pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
    #     ang_coef=1 / np.pi,
    #     lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
    #     ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL)

    env = rlgym_sim.make(
        tick_skip=tick_skip,
        team_size=team_size,
        spawn_opponents=spawn_opponents,
        terminal_conditions=terminal_conditions,
        reward_fn=reward_fn,
        obs_builder=obs_builder,
        action_parser=action_parser,
    )

    return env


if __name__ == "__main__":
    from rlgym_ppo import Learner
    import os

    root_folder = "checkpoints/"
    checkpoint_load_folder = ""

    # Get a list of all subdirectories in the root folder
    subdirectories = [
        f
        for f in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, f))
    ]

    highest_checkpoint_folder = None
    highest_checkpoint_number = -1

    # Iterate through the subdirectories to find the highest "checkpoints-" directory
    for subdir in subdirectories:
        if subdir.startswith("checkpoints-") and subdir[12:].isdigit():
            checkpoint_number = int(subdir[12:])
            if checkpoint_number > highest_checkpoint_number:
                highest_checkpoint_number = checkpoint_number
                highest_checkpoint_folder = subdir

    if highest_checkpoint_folder:
        checkpoint_load_folder = os.path.join(root_folder, highest_checkpoint_folder)

        # Now, let's find the highest numbered folder within the highest checkpoint folder
        highest_numbered_subfolder = None
        highest_number = -1
        checkpoint_subdirectories = [
            f
            for f in os.listdir(checkpoint_load_folder)
            if os.path.isdir(os.path.join(checkpoint_load_folder, f))
        ]

        for subdir in checkpoint_subdirectories:
            if subdir.isdigit():
                subdir_number = int(subdir)
                if subdir_number > highest_number:
                    highest_number = subdir_number
                    highest_numbered_subfolder = subdir

        if highest_numbered_subfolder:
            checkpoint_load_folder = os.path.join(
                checkpoint_load_folder, highest_numbered_subfolder
            )
    else:
        print("No 'checkpoints-' directories found in the checkpoints folder.")

    metrics_logger = ExampleLogger()

    # number of instances of the environment to run in parallel
    n_proc = 75

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(
        build_rocketsim_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        metrics_logger=metrics_logger,
        ppo_batch_size=500_000,
        ts_per_iteration=1_000_000,
        exp_buffer_size=1_000_000,
        ppo_minibatch_size=50_000,
        ppo_ent_coef=0.001,
        ppo_epochs=2,
        standardize_returns=True,
        standardize_obs=False,
        save_every_ts=100_000,
        checkpoints_save_folder="checkpoints/checkpoints",
        **(
            {"checkpoint_load_folder": checkpoint_load_folder}
            if checkpoint_load_folder != ""
            else {}
        ),
        timestep_limit=1_000_000_000,
        log_to_wandb=True,
    )
    learner.learn()
