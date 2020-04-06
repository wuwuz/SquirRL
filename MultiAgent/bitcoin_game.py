import random
import numpy as np
import argparse
import ray
from gym import spaces
from ray.tune.registry import register_env
from ray.rllib.models.preprocessors import get_preprocessor
from ray import tune
from ray.rllib.agents.pg.pg import PGTrainer
from ray.rllib.agents.pg.pg_policy import PGTFPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import try_import_tf
from ray.tune.util import flatten_dict
from ray.tune.result import (NODE_IP, TRAINING_ITERATION, TIME_TOTAL_S,
                             TIMESTEPS_TOTAL, EXPR_PARAM_FILE,
                             EXPR_PARAM_PICKLE_FILE, EXPR_PROGRESS_FILE,
                             EXPR_RESULT_FILE)
from functools import reduce
from itertools import (chain, takewhile)
from ray.rllib.agents.ppo import PPOTrainer             
#from OSM import OSM
import os
import csv
import pandas as pd
import math
import time
import constants
import mdptoolbox

from OSM import OSM
from BitcoinEnv import BitcoinEnv
os.environ['https_proxy'] = ' '


# to calculate the episode stats
def on_episode_start(info):
    episode = info["episode"]

def on_episode_step(info):
    episode = info["episode"]
    i = 0
    while True:
        last_info = episode.last_info_for(str(i))
        if last_info is None:
            break
        if last_info is not None:
            if "Won blocks" in last_info:
                won = last_info["Won blocks"]
                total = last_info["Total blocks"]
            else:
                total = 0
        else:
            total = 0
    
        if total > 0:
            relative_reward = float(won)/float(total)
        else:
            relative_reward = 0
        episode.user_data["rel_reward_" + str(i)] = relative_reward
        for k in range(6):
            if '0' in last_info:
                episode.user_data[str(k) + '_' + str(i)] = last_info[str(k)]
            else:
                episode.user_data[str(k) + '_' + str(i)] = 0
        i += 1

def on_episode_end(info):
    episode = info["episode"]
    total = 0
    i = 0
    while True:
        if "rel_reward_" + str(i) not in episode.user_data:
            break
        relative_reward = episode.user_data["rel_reward_" + str(i)]
        episode.custom_metrics["relative_reward_" + str(i)] = relative_reward
        for k in range(6):
            episode.custom_metrics['act' + str(k) + '_' + str(i)] = episode.user_data[str(k) + '_' + str(i)]
        total += relative_reward
        i += 1
    episode.custom_metrics["total_relative_reward"] = total
class OSM_strategy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.osm = OSM(config['alpha'], config['gamma'], config['blocks'])
        self.osm.MDP_matrix_init()
        P, R = self.osm.get_MDP_matrix()
        solver = mdptoolbox.mdp.PolicyIteration(P, R, 0.99)
        solver.run()
        self.blocks = config['blocks']
        self.optimal_policy = solver.policy
    def OSM_act(self, s):
        curr_s = list(s)
        if s[3] == constants.NORMAL:
            curr_s[3] = 'normal'
        elif s[3] == constants.FORKING:
            curr_s[3] = 'forking'
        else:
            curr_s[3] = 'catch up'
        smaller_state = curr_s[:2] + [curr_s[3]]
        smaller_state = tuple(smaller_state)
        if curr_s[0] >= self.blocks or curr_s[1] >= self.blocks:
            if curr_s[0] > curr_s[1]:
                return 1
            else:
                return 0
        if smaller_state in self.osm._state_dict:
            return self.optimal_policy[self.osm._name_to_index(smaller_state)]
        else:
            return 2
    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        actions = []
        for i in range(len(obs_batch)):
            actions.append(self.OSM_act(obs_batch[i]))
        return actions, [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
# Train an RL agent against an OSM agent.  0 agent is OSM, 1 agent is RL
def run_OSM(args):
    def select_policy(agent_id):
        return agent_id
    mine_act_space = spaces.Discrete(6)

    # define the state space, one for parties that have access to spy info and one without
    spy_state_space = constants.make_spy_space(len(args.alphas), args.blocks)
    blind_state_space = spaces.Tuple((spaces.Discrete(args.blocks + 3),
                    spaces.Discrete(args.blocks + 3),
                    spaces.Discrete(args.blocks + 3),
                    spaces.Discrete(3)))
    policies = dict()
    policies['0'] = (OSM_strategy, blind_state_space, mine_act_space, {'alpha': args.alphas[0], 'gamma': args.gammas[0], 'blocks': args.blocks})
    if args.spy[1] == 1:
        policies['1'] = (None, spy_state_space, mine_act_space, {
                    "model": {
                        "use_lstm":args.use_lstm
                    }
                })
    else:
        policies['1'] = (None, blind_state_space, mine_act_space, {
                        "model": {
                            "use_lstm":args.use_lstm
                        }
                    })
    register_env(
        "bitcoin_vs_OSM",
        lambda config: BitcoinEnv(config)
    )
    env_config = {'max_hidden_block': args.blocks,
            'alphas': args.alphas,
            'gammas': args.gammas,
            'ep_length': args.ep_length,
            'print': False,
            'spy': args.spy,
            'team_spirit': args.team_spirit
        }
    tune.run(
        args.trainer,
        #loggers = [CustomLogger], 
        stop={"episodes_total": args.episodes},
        config={
            "env": "bitcoin_vs_OSM",
            #"vf_share_layers": True,
            "gamma": 0.99,
            "num_workers": args.workers,
            "num_envs_per_worker": 1,
            "batch_mode": "complete_episodes",
            "train_batch_size": args.workers*args.ep_length,
            "entropy_coeff": .5,
            "entropy_coeff_schedule": args.ep_length*args.episodes,
            "multiagent": {
                "policies_to_train": ["1"],
                "policies": policies,
                "policy_mapping_fn": select_policy,
            },
            "env_config": env_config,
            "callbacks": {
                "on_episode_start": on_episode_start,
                "on_episode_step": on_episode_step,
                "on_episode_end": on_episode_end
            }
        },
        keep_checkpoints_num=1,
        checkpoint_freq=3) 
def run_saved_OSM(args):
    ray.init()
    # define the action space
    mine_act_space = spaces.Discrete(6)

    # define the state space, one for parties that have access to spy info and one without
    spy_state_space = constants.make_spy_space(len(args.alphas), args.blocks)
    blind_state_space = spaces.Tuple((spaces.Discrete(args.blocks + 3),
                    spaces.Discrete(args.blocks + 3),
                    spaces.Discrete(args.blocks + 3),
                    spaces.Discrete(3)))
    print("Testing OSM", args.alphas)
    def select_policy(agent_id):
        return agent_id
    policies_to_train = ["1"]
    policies = dict()
    policies['0'] = (OSM_strategy, blind_state_space, mine_act_space, {'alpha': args.alphas[0], 'gamma': args.gammas[0], 'blocks': args.blocks})
    if args.spy[1] == 1:
        policies['1'] = (None, spy_state_space, mine_act_space, {
                    "model": {
                        "use_lstm":args.use_lstm
                    }
                })
    else:
        policies['1'] = (None, blind_state_space, mine_act_space, {
                        "model": {
                            "use_lstm":args.use_lstm
                        }
                    })
    env_config={"max_hidden_block":args.blocks,
        "alphas":args.alphas,
        "gammas":args.alphas,
        'ep_length':args.ep_length,
        'print': args.debug,
        'spy': args.spy,
        'team_spirit': args.team_spirit}
    trainer = PPOTrainer(env=BitcoinEnv, config={
                "num_workers": 0,
                "multiagent": {
                    "policies_to_train": policies_to_train,
                    "policies": policies,
                    "policy_mapping_fn": select_policy,
                },
                "env_config": env_config
            })
    env = BitcoinEnv(env_config=env_config)
    trainer.restore(args.save_path)
    prep = get_preprocessor(blind_state_space)(blind_state_space)
    loaded_policies = dict()
    for k in range(len(args.alphas)):
        loaded_policies[str(k)] = trainer.get_policy(str(k))
    trials = 10000
    rel_rewards = []
    OSM_rewards = []
    for i in range(trials):
        obs = env.reset()
        isDone = False
        while not isDone:
            action_dict = dict()
            for k in range(len(policies)):
                action_dict[str(k)], _, _ = loaded_policies[str(k)].compute_single_action(obs=prep.transform(obs[str(k)]), state = [])
            obs, _, done, _ = env.step(action_dict)
            isDone = done['__all__']
        selfish = env._accepted_blocks[1]
        OSM = env._accepted_blocks[0]
        total_blocks = np.sum(env._accepted_blocks)
        rel_rewards.append(float(selfish)/total_blocks)
        OSM_rewards.append(float(OSM)/total_blocks)
        if i % 100 == 0:
            print("RL:", np.mean(rel_rewards), "OSM:", np.mean(OSM_rewards))
    print(str(j) + '\t' + str(np.mean(OSM_rewards)) + '\t' + str(np.mean(rel_rewards)), file=open('output.txt', "a"))
# run saved strategies from an RL^k game
def run_saved(args):
    ray.init()
    # define the action space
    mine_act_space = spaces.Discrete(6)

    # define the state space, one for parties that have access to spy info and one without
    spy_state_space = constants.make_spy_space(len(args.alphas), args.blocks)
    blind_state_space = spaces.Tuple((spaces.Discrete(args.blocks + 3),
                    spaces.Discrete(args.blocks + 3),
                    spaces.Discrete(args.blocks + 3),
                    spaces.Discrete(3)))
    print("Testing alpha", args.alphas)
    def select_policy(agent_id):
        return agent_id
    policies_to_train = [str(i) for i in range(len(args.alphas))]
    policies = dict()
    for i in range(len(args.alphas)):
        if args.spy[i] == 1:
            policies[str(i)] = (None, spy_state_space, mine_act_space, {
                    "model": {
                        "use_lstm":args.use_lstm

                    }
                })
        else:
            policies[str(i)] = (None, blind_state_space, mine_act_space, {
                    "model": {
                        "use_lstm":args.use_lstm
                    }
                })
    env_config={"max_hidden_block":args.blocks,
        "alphas":args.alphas,
        "gammas":args.alphas,
        'ep_length':args.ep_length,
        'print': args.debug,
        'spy': args.spy,
        'team_spirit': args.team_spirit}
    trainer = PPOTrainer(env=BitcoinEnv, config={
                "num_workers": 0,
                "multiagent": {
                    "policies_to_train": policies_to_train,
                    "policies": policies,
                    "policy_mapping_fn": select_policy,
                },
                "env_config": env_config
            })
    env = BitcoinEnv(env_config=env_config)
    trainer.restore(args.save_path)
    prep = get_preprocessor(spy_state_space)(spy_state_space)
    loaded_policies = dict()
    for k in range(len(args.alphas)):
        loaded_policies[str(k)] = trainer.get_policy(str(k))
    trials = 10000
    rel_rewards = []
    for i in range(trials):
        obs = env.reset()
        isDone = False
        while not isDone:
            action_dict = dict()
            for k in range(len(policies)):
                action_dict[str(k)], _, _ = loaded_policies[str(k)].compute_single_action(obs=prep.transform(obs[str(k)]), state = [])
            obs, _, done, _ = env.step(action_dict)
            isDone = done['__all__']
        selfish = np.sum(env._accepted_blocks[:-1])
        total_blocks = np.sum(env._accepted_blocks)
        rel_rewards.append(float(selfish)/total_blocks)
        if i % 100 == 0:
            print("Current mean:", np.mean(rel_rewards))
    print(str(j) + '\t' + str(np.mean(rel_rewards)), file=open('output.txt', "a"))
def run_RL(args):
    def select_policy(agent_id):
        return agent_id
    policies_to_train = [str(i) for i in range(len(args.alphas))]
    
    policies = dict()

    # define the action space
    mine_act_space = spaces.Discrete(6)

    # define the state space, one for parties that have access to spy info and one without
    spy_state_space = constants.make_spy_space(len(args.alphas), args.blocks)
    blind_state_space = spaces.Tuple((spaces.Discrete(args.blocks + 3),
                    spaces.Discrete(args.blocks + 3),
                    spaces.Discrete(args.blocks + 3),
                    spaces.Discrete(3)))

    for i in range(len(args.alphas)):
        if args.spy[i] == 1:
            policies[str(i)] = (None, spy_state_space, mine_act_space, {
                    "model": {
                        "use_lstm":args.use_lstm

                    }
                })
        else:
            policies[str(i)] = (None, blind_state_space, mine_act_space, {
                    "model": {
                        "use_lstm":args.use_lstm
                    }
                })
    register_env(
        "bitcoin_team" + str(args.team_spirit * 10),
        lambda config: BitcoinEnv(config)
    )
    if args.debug:
        env_config = {'max_hidden_block': args.blocks,
            'alphas': args.alphas,
            'gammas': args.gammas,
            'ep_length': args.ep_length,
            'print': True,
            'spy': args.spy,
            'team_spirit': args.team_spirit
        } 
        tune.run(
            args.trainer,
            stop={"episodes_total": args.episodes},
            config={
                "env": "bitcoin_team" + str(args.team_spirit * 10),
                #"vf_share_layers": True,
                "gamma": 0.99,
                "num_workers": args.workers,
                "num_envs_per_worker": 1,
                "batch_mode": "complete_episodes",
                "train_batch_size": args.workers*args.ep_length,
                "entropy_coeff": .5,
                "entropy_coeff_schedule": args.ep_length*args.episodes,
                "multiagent": {
                    "policies_to_train": policies_to_train,
                    "policies": policies,
                    "policy_mapping_fn": select_policy,
                },
                "env_config": env_config,
            })
    else:
        env_config = {'max_hidden_block': args.blocks,
            'alphas': args.alphas,
            'gammas': args.gammas,
            'ep_length': args.ep_length,
            'print': False,
            'spy': args.spy,
            'team_spirit': args.team_spirit
        }
        tune.run(
            args.trainer,
            #loggers = [CustomLogger], 
            stop={"episodes_total": args.episodes},
            config={
                "env": "bitcoin_team" + str(args.team_spirit * 10),
                #"vf_share_layers": True,
                "gamma": 0.99,
                "num_workers": args.workers,
                "num_envs_per_worker": 1,
                "batch_mode": "complete_episodes",
                "train_batch_size": args.workers*args.ep_length,
                "entropy_coeff": .5,
                "entropy_coeff_schedule": args.ep_length*args.episodes,
                "multiagent": {
                    "policies_to_train": policies_to_train,
                    "policies": policies,
                    "policy_mapping_fn": select_policy,
                },
                "env_config": env_config,
                "callbacks": {
                    "on_episode_start": on_episode_start,
                    "on_episode_step": on_episode_step,
                    "on_episode_end": on_episode_end
                }
            },
            checkpoint_score_attr="episode_reward_mean",
            keep_checkpoints_num=1,
            checkpoint_freq=3)    
def main():
    CLI = argparse.ArgumentParser()
    # hash power of each player
    CLI.add_argument("--alphas", nargs = '*', type = float, default = [.26, .26])
    # follower fraction of each player
    CLI.add_argument('--gammas',nargs='*',type=float,default = [0, 0])
    # the number of hidden blocks allowed until players must play honestly
    CLI.add_argument('--blocks', type = int, default = 5)
    # which algo to use
    CLI.add_argument('--trainer', type = str, default = 'PPO')
    CLI.add_argument('--episodes', type = int, default = 100000)
    CLI.add_argument('--ep_length', type = int, default = 100)
    # give particular players the ability to see the hidden states of other players
    CLI.add_argument('--spy', nargs='*', type=int, default = [1, 1])
    # do the OSM experiment; currently only supported for RL vs OSM, can be expanded if need be
    CLI.add_argument('--OSM', type = bool, default = False)
    CLI.add_argument('--workers', type = int, default = 7)
    # how much to value the team (1 is fully support team, 0 is lone wolf.)
    CLI.add_argument('--team_spirit', type = float,default = 1.)
    # print the blockchain graphs at each step to see what's going on
    CLI.add_argument('--debug', type = bool, default = False)
    # input the global path of the checkpoint you want to run an evaluation on
    CLI.add_argument('--save_path', type = str, default = '')
    CLI.add_argument('--use_lstm', type = str, default = False)
    args = CLI.parse_args()   
    
    if args.OSM:
        args.spy = [0 for i in range(len(args.alphas))]
        if len(args.save_path) > 0:
            run_saved_OSM(args)
        else:
            run_OSM(args)
    else:
        if len(args.save_path) > 0:
            run_saved(args)
        else:
            run_RL(args)

if __name__ == "__main__":
    main()