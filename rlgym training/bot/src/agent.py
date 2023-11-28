import os
import numpy as np
from rlgym_ppo import ppo
from rlgym_ppo.ppo import ContinuousPolicy
import torch

model_zip = "exit_save.zip"


class Agent:
    def __init__(self):
        # Get the directory of agent.py
        script_dir = os.path.dirname(os.path.realpath(__file__))

        root_folder = os.path.join(script_dir, "../../checkpoints/")
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
            checkpoint_load_folder = os.path.join(
                root_folder, highest_checkpoint_folder
            )

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

        input_shape = 109  # obs_space_size = np.prod(obs_space_size)
        output_shape = 16  # 2*num_actions_in_rocket_league
        layer_sizes = (256, 256, 256)  # (256,256,256) by default
        self.policy = ContinuousPolicy(input_shape, output_shape, layer_sizes, "cpu")
        self.policy.load_state_dict(
            torch.load(os.path.join(checkpoint_load_folder, "PPO_POLICY.pt"))
        )
        # self.model =

    def act(self, obs):
        action, logprob = self.policy.get_action(obs, deterministic=False)
        # action = self.policy.eval(obs)
        # action, _ = self.model.predict(obs)
        return action.tolist()
