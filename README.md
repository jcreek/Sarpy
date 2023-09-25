# Sarpy

## Prerequisites

1. Install Python version 3.9.0
2. Install Visual C++ 14.0 or greater from https://visualstudio.microsoft.com/visual-cpp-build-tools/
3. Run `pip install setuptools==65.5.0 pip==21` as gym 0.21 installation is broken with more recent versions
4. You can now install the packages manually or using the requirements.txt file. To do the latter, run `pip3 install -r requirements.txt`

### Manual package installation

You will likely see errors during installations here. Largely these should be ignored, just try running things after installing, then work out manually what is broken. There are lots of deprecated or broken packages involved in running rlgym unfortunately, as it does not support the newer version 2 of stable baselines 3 that uses gymnasium instead of gym.

1. Run `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`
2. Run `pip3 install "stable-baselines3[extra]==1.8.0"`
3. Run `pip3 install "gym[box2d]"`

## Other useful commands

`tensorboard --logdir=logs` in a new terminal to load the web UI to track agent training.

`pip3 freeze > requirements.txt` to save current packages installed into an updated requirements.txt file.
