# Introduction
These are instructions on how to start reproducing the (multiagent) experiments in the paper.
To start, use Python 3.7.13, and install the packages according to the requirements.txt file.  Using a virtualenv is recommended.

# Training RL vs RL

We will have a game between a training RL agent and a RL agent, each with 40% hash power.

In terminal type

    python3 bitcoin_game.py --OSM 0 --alphas .4 .4 --gammas 0 0 --players 2

--OSM declares how many OSM agents we have.  --alphas is a list for the hash power
of each strategic player.  --players is the number of strategic players.  --gammas is a list for how much follow fraction each strategic player has.

You should see some output on how the agent is performing.  "Reward" being higher means the strategic agents are performing better.

For other options, you can check bitcoin_game.py and look at the comments in the command line arguments.

Note that given the batchsize in training is so high, you might find each training step takes a while.  The reason batchsize is so high is because in RL the variance of updates is really high.  Check likes 440 and 441 along with documentation for RLLib (version 0.8.5) to change batchsize for your own needs.

You can customize the experiment using the command line arguments to run other experiments.

Some other examples as seen in our paper:

# Training RL vs OSM

We will have a game between a training RL agent and an OSM agent, each with 40% hash power.

In terminal type

    python3 bitcoin_game.py --OSM 1 --alphas .4 .4 --gammas 0 0 --players 2


# Training RL vs RL vs RL

We will have a game between three RL agents, each with 30% hash power.

In terminal type

    python3 bitcoin_game.py --OSM 0 --alphas .3 .3 .3 --gammas 0 0 0 --players 3