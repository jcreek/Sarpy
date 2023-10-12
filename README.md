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

`tensorboard --logdir=out/logs --bind_all` in a new terminal to load the web UI to track agent training. `--bind_all` is optional and exposes it on the network so you can monitor it from another device.

`pip3 freeze > requirements.txt` to save current packages installed into an updated requirements.txt file.

## Training

Run the python file rl-training.py

## Using the bot

Within the bot folder, run the python file run_gui.py

## Training plan

This bot is configured to be able to play any Rocket League game mode. The plan for training it is as below. This will be done by playing games in as many instances of Rocket League simultaneously as possible.

### Stage 1 (in progress)

Learn the basic mechanics as fast as possible. To do this I'm using 3v3, so that there are 6 agents training for each instance of the game. This will make the logs noisier, but should give the bot more exposure to the basics of the game. 

#### 0 steps

Reward functions at this point are as below, along with their scale:

- VelocityPlayerToBallReward - 0.1
- VelocityBallToGoalReward - 1
- LiuDistanceBallToGoalReward - 1 
- EventReward - 1
    - team_goal=100.0,
    - concede=-100.0,
    - shot=5.0,
    - save=30.0,
    - demo=10.0,
- BallYCoordinateReward - 0.1
- RewardIfClosestToBall - 0.2 
- LiuDistancePlayerToBallReward - 0.1 
- FaceBallReward - 0.2 
- TouchBallReward - 1
- AlignBallGoal - 0.1
- KickoffReward - 10

The primary goal is to get the model to learn to kickoff, and generally aim to be ball chasing, with a plan to get the ball into the opposing goal.

Terminal conditions for the first 265million steps: 

- TimeoutCondition(fps * 30)
- NoTouchTimeoutCondition(fps * 10)
- GoalScoredCondition()

#### ~250 million steps

After 265 million steps I changed the terminal conditions and reward weightings. 

- TimeoutCondition -> fps * 300

- VelocityPlayerToBallReward -> 1
- TouchBallReward -> 10
- LiuDistanceBallToGoalReward -> 50

This is to try to get the bot to gain experience in other areas of the game now it has vaguely got the hang of what a kickoff is.

#### ~500 million steps

After 528 million steps I changed things again, changing the rewards as below: 

- VelocityBallToGoalReward -> 10
- EventReward -> 10
- RewardIfClosestToBall -> 1
- LiuDistancePlayerToBallReward -> 1
- AlignBallGoal -> 20

#### ~1 billion steps

The bot should now have acceptable mechanics and a vague understanding of 3v3 strategies.

However, I noticed that there's a lot of dead time, where it's not necessarily driving towards the ball or preparing to defend or receive a pass, so I changed things again, changing the rewards as below: 

- VelocityPlayerToBallReward -> 50
- LiuDistanceBallToGoalReward -> 20
- RewardIfClosestToBall -> 10

### Stage 2

Learn the strategies for playing each game mode. To do this I will make the bot train in each game mode in series, or possibly in parallel if I can configure it to run different game modes in different instances of Rocket League. 

Reward functions?

### Stage 3

Learn more advanced mechanics.

### Stage 4

Test it. 

### Stage 5

Test it in a bot tournament.
