
# SquirRL

This is the implementation of the paper *SquirRL: Automating Attack Discovery on Blockchain Incentive Mechanisms with Deep Reinforcement Learning* (https://arxiv.org/abs/1912.01798). 

## Single Agent Part

We test 3 different chain rules: Bitcoin(longest chain rule), Ethereum(uncle block rule) and GHOST (heaviest sub-tree rule). 

### Basic Usage

In `\SingleAgent`,  run

```
python btc.py
python eth.py
python ghost.py
```

and you can train the RL agents and test them. 

### Main Files

`\SingleAgent\environment.py` 

The interactive environment for 3 different protocols. OpenAI gym type APIs. 

`\SingleAgent\btc.py` , `\SingleAgent\eth.py`, `\SingleAgent\ghost.py` 

The training and testing part of the RL agent for the Bitcoin/Ethereum/GHOST protocols. You can find more detailed comments of training in `\SingleAgent\btc.py`. 

`\SingleAgent\optimal_policy.py`

The implementation of *Optimal Selfish Mining*. We saved the policy in `optimal_policy.txt`. 

## Block Withholding Part

We reproduce the two-agent block withholding result in [The Miner's Dilemma](https://arxiv.org/abs/1411.7099). Given the proportion of mining power two mining pools possess, there exists a unique Nash equilibirium which they fall in when deciding how much mining power to infiltrate in each other's pool. We use the *SquirRL* framework to find such equilibrium.

### Basic Usage
In `\BlockWithholding`, run
```
python blockwithholding.py > result.txt
```
to train the RL agents and find the strategies and rewards of them.

You can specify the mining power of the two agents via argument `alphas`, e.g., for two mining pools possessing 30% and 50% mining power respectively, you can run the program with `--alphas 0.3 0.5`.

The policies and rewards of our training result is saved in `\result\result.csv`.


## Contributor

Ordered by alphabet: 

Charlie Hou charlieh@andrew.cmu.edu

Mingxun Zhou : zhoumingxun@pku.edu.cn

Yan Ji : yj348@cornell.edu


