# Introduction
These are instructions on how to start reproducing the (multiagent) experiments in the paper.
To start, you should have all the packages that come with anaconda, as well as Ray (ver. 0.7.6), OpenAI Gym, and mdptoolbox.  

The workflow generally goes as follows:

(1) Train your agent

(2) Evaluate a saved agent to get reliable metrics on the strategy + see qualitatively what strategy is learned.

In general, your command line arguments will be the same for both steps, with --save_path being the exception (the file you're loading strategies from during the evaluation phase).

We will go through a few examples now.

# Training RL vs OSM

We will have a game between a training RL agent and an OSM agent, each with 40% hash power and zero cooperation.

In terminal type

    python3 bitcoin_game.py --OSM True --alphas .4 .4 --gammas 0 0 --team_spirit 0 

In the terminal, you should get some output about the performance of your algorithm.  When you feel that your agent has has finished learning, you can evaluate it.

In terminal type

    python3 bitcoin_game.py --OSM True --alphas .4 .4 --gammas 0 0 --save_path your_save_path --debug True

Notice that 'your_save_path' may vary.  For instance, mine was /afs/ece.cmu.edu/usr/charlieh/ray_results/PPO/PPO_bitcoin_team10.0_0_2020-04-05_14-41-31i7wz5afh/checkpoint_249/checkpoint-249.  

## Debug mode
Notice I used the --debug option here.  This can be used for training or evaluation, it simply just outputs visualizations of the ongoing game.  This is helpful for debugging or seeing the strategies play out live.  There is a lot of output that can come as a result of this.  Note that with many workers, you can have output that doesn't make sense for threading reasons, so try to keep it single-process if you want to analyze the output.

Some example output looks like this 

    Step 88
    Agent 0 action Adopt Team
    Agent 1 action Adopt
    Rewards [0, 0, 1]
    Block receiver 0
    Obs {'0': (1, 0, 1, 0, 0, 0, 0, 0, 0), '1': (0, 0, 0, 0, 0, 0, 0, 0, 0)}


    (2, [0, 1, 2], 0, [0, 0, 0]) 


    Step 89
    Agent 0 action Publish all
    Agent 1 action Adopt Team
    Rewards [1, 0, 0]
    Block receiver 2
    Obs {'0': (0, 1, 0, 0, 0, 0, 0, 0, 0), '1': (0, 1, 0, 0, 0, 0, 0, 0, 0)}


    (0, [0, 1, 2], 0, [0, 0, 0]) ─ (2, [2], 1, [0, 0, 1]) 

The first 4 liness denote the actions taken by each agent, who got what reward (the last one in the list is the honest party), and block receiver denotes which party just solved the proof of work puzzle.  Obs shows what each agent sees:

(Length of chain being mined on by agent, length of longest public chain, how many blocks are owned by the agent on the chain it is mining on, the current forking characteristic, length of chain being mined on by agent 0, length of chain being mined on by agent 1, number of blocks owned by the current agent by agent 0, number of blocks owned by the current agent by agent 1, number of blocks owned by the current agent by agent 2)

By convention, the last agent is always the honest party.

The last output, which is a tree, shows the structure of the currently relevant blockchain (everything before a consensus is irrelevant).  The tree grows rightwards.  The tuples for each node represent:

(agent who mined it, the agents that acknowledge this block, how far from the last universally acknowledged block this block is, how much of this chain is owned by each party)

Note that the first one may have '[0,0,0]' despite being owned by someone.  This is because it was counted as part of the last reward distribution.


# Training RL vs RL vs RL vs RL

This will be a game between 4 RL agents with 20% hash power, zero cooperation, and no spy info.

In terminal type

    python3 bitcoin_game.py --alpha .2 .2 .2 .2 --gammas 0 0 0 0 --spy 0 0 0 0 --team_spirit 0

Once again, you can evaluate using

    python3 bitcoin_game.py --alpha .2 .2 .2 .2 --gammas 0 0 0 0 --spy 0 0 0 0 --save_path your_save_path

# Cooperation

This will be a cooperative game between 2 RL agents with .26 hash power and 100% cooperation.  (You need spy information to cooperate properly, so the option is enabled here).

In terminal type 

    python3 bitcoin_game.py --alpha .26 .26 --gammas 0 0 --spy 1 1 --team_spirit 1

And you can evaluate using

    python3 bitcoin_game.py --alpha .26 .26 --gammas 0 0 --spy 1 1 --team_spirit 1 --save_path your_save_path


These are the first basic examples to replicate some experiments.  Please read the main method of bitcoin_game.py to see the other things you can change.  Note that the arguments for alpha, gammas, and spy have to have the same length.  

# Your own experiments

You may want to try out different configurations of agents of different policies and such.  If so, you can look at bitcoin_game.py and use the run_* methods to model your experiments after, and run them.  