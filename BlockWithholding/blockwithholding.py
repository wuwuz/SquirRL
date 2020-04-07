from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import argparse
from gym import spaces
import ray
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
from ray.tune.logger import pretty_print

import os
import csv
import mdptoolbox
import pandas as pd
import math
import sympy as sym

tf = try_import_tf()

CLI = argparse.ArgumentParser()

CLI.add_argument(
    "--alphas",
    nargs = '*',
    type = float,
    default = [.4, .5]
)
CLI.add_argument(
    "--impact",
    type = float,
    default = .2
)
CLI.add_argument(
    "--threshold",
    type = float,
    default = .02
)
CLI.add_argument(
    '--algo',
    type = str,
    default = 'PPO'
)
CLI.add_argument(
    '--use_lstm',
    type = bool,
    default = False
)
CLI.add_argument(
    '--gamma',
    type = float,
    default = 0.99
)
CLI.add_argument(
    '--lr',
    type = float,
    default = 1e-6
)
CLI.add_argument(
    '--lmbda',
    type = float,
    default = 1.0
)
CLI.add_argument(
    '--iteration',
    type = int,
    default = 10
)
CLI.add_argument(
    '--episodes',
    type = int,
    default = 1e6
)
CLI.add_argument(
    '--ep_length',
    type = int,
    default = 1
)
CLI.add_argument(
    '--gpus',
    type = int,
    default = 0
)
CLI.add_argument(
    '--NE',
    type = bool,
    default = False
)
CLI.add_argument(
    '--workers',
    type = int,
    default = 5
)
CLI.add_argument(
    '--evaluate',
    type = bool,
    default = False
)
CLI.add_argument(
    '--eval_ep',
    type = int,
    default = 1
)
args = CLI.parse_args()

eps = 1e-6

#setting in miner's dilemma
ACTION_SPACE = spaces.Box(low=np.array([0.]), high=np.array([1.]), dtype=np.float32)
STATE_SPACE = spaces.Discrete(1)
NE = dict()

def get_optimal_strategy(a, b, y):
    x = sym.Symbol('x', real=True)
    R1 = (a - x) / (1. - x - y)
    R2 = (b - y) / (1. - x - y)
    r1 = ((b * R1) + x * (R1 + R2)) / (a * b + a * x + b * y)
    d1 = sym.Eq(sym.diff(r1, x), 0.)

    A = sym.solve(d1, x)
    
    if A:
        for i in A:
            if (i > eps and i < a - eps):
                return i, r1.subs(x, i)

    if (a * b + b * y < eps or r1.subs(x, 0.) > r1.subs(x, a) - eps):
        return 0., r1.subs(x, 0.)
    else:
        return a, r1.subs(x, a)

def plot_Nash_equilibrium(x, y, z, name):
    x, y = np.meshgrid(x,y)

    z = z.transpose()
    intensity = z.reshape(len(y), len(x))
   
    plt.title(name)
    plt.pcolormesh(x, y, intensity, rasterized=True)
    plt.clim(0., 1.2)
    plt.colorbar() #need a colorbar to show the intensity scale
    #plt.show() #boom
   
def compute_reward(a, b, x, y):
    if (x + y > 1 - eps):
        return {'0': 0., '1': 0.}
    if (y < eps and a < eps):
        return {'0': 1., '1': 1.}
    if (x < eps and b < eps):
        return {'0': 1., '1': 1.}
    R1 = (a - x) / (1. - x - y)
    R2 = (b - y) / (1. - x - y)
    r1 = ((b * R1) + x * (R1 + R2)) / (a * b + a * x + b * y)
    r2 = ((a * R2) + y * (R1 + R2)) / (a * b + a * x + b * y)
    return {'0': r1, '1': r2}

def get_Nash_equilibrium(alphas):
    a = alphas[0]
    b = alphas[1]
    if (a + b > 1. or (a < eps and b < eps)): 
        return 0., 0., 1., 1.

    x = 0.
    y = 0.
    while (True):
        X, R1 = get_optimal_strategy(a, b, y)
        Y, R2 = get_optimal_strategy(b, a, x)
        
        if (abs(X - x) < eps and abs(Y - y) < eps):
            rev = compute_reward(a, b, x, y)
            return x, y, rev['0'], rev['1']
        
        x = X
        y = Y

class MigrationEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.action_space = ACTION_SPACE
        self.observation_space = STATE_SPACE

        self.HASHRATE = np.array(env_config['alphas'])
        self.alphas = np.array(env_config['alphas'])
        self.N = len(self.alphas)
        self.episode_length = env_config['ep_length']
        self.attr = np.full((self.N), 1.)
        self.impact = args.impact
        self.threshold = args.threshold
        self.largest_pool = np.full((self.N, 2), -1)

        self.num_moves = 0
   
    def compute_states(self):
        obs_state = dict()
        self.largest_pool = np.full((self.N, 2), -1)
        for i in range(len(self.alphas)):
            tmp = np.array([self.alphas[i], 0., 0., 0.])
            rest = []
            
            for j in range(len(self.alphas)):
                if i == j:
                    continue
                if self.alphas[j] >= tmp[1]:
                    if (self.largest_pool[i][1] > -1):
                        rest.append(tmp[2])
                    tmp[2] = tmp[1]
                    self.largest_pool[i][1] = self.largest_pool[i][0]
                    tmp[1] = self.alphas[j]
                    self.largest_pool[i][0] = j
                elif self.alphas[j] > tmp[2]:
                    if (self.largest_pool[i][1] > -1):
                        rest.append(tmp[2])
                    tmp[2] = self.alphas[j]
                    self.largest_pool[i][1] = j
                else: 
                    rest.append(self.alphas[j])

            tmp[3] = np.array(rest).std()
            obs_state[str(i)] = tmp
        return obs_state

    #reset the environment to the starting state
    def reset(self):
        self.num_moves = 0
        self.alphas = np.array(self.HASHRATE)
        self.attr = np.full((self.N), 1.)
        return self.compute_states() 

    def construct_action(self, action_dict):
        action = np.empty([self.N, self.N], dtype=np.float32)
        for i in range(self.N):
            action[i] = np.full((self.N), self.alphas[i] * action_dict[str(i)][2])
            action[i][i] = 0.
            if self.largest_pool[i][0] > -1:
                action[i][self.largest_pool[i][0]] = self.alphas[i] * action_dict[str(i)][0]
            if self.largest_pool[i][1] > -1:
                action[i][self.largest_pool[i][1]] = self.alphas[i] * action_dict[str(i)][1]
            if (action[i].sum() > 1 - eps):
                action[i] = action[i] / (action[i] + eps)
        
        return action

    
    def step(self, action_dict):
        self.num_moves += 1

        a = np.empty([self.N, self.N], dtype=np.float32)
        b = np.empty([self.N], dtype=np.float32)

        action = self.construct_action(action_dict)
        #print("states:{}\n{}\n{}\n".format(self.compute_states(), action_dict, action))
        infiltrate = action.sum(1)
        infiltrated = action.sum(0)
        total = action.sum()

        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    a[i][j] = self.alphas[i] + infiltrated[i]
                else:
                    a[i][j] = -action[i][j]
            b[i] = (self.alphas[i] - infiltrate[i]) / (1 - total)

        r = np.empty([self.N], dtype=np.float32)
        try:
            r = np.linalg.solve(a, b)
        except(RuntimeError, np.linalg.LinAlgError):
            r = np.full((self.N), 1.)
        R = dict()
        for i in range(self.N):
            R[str(i)] = r[i]

        done = {"__all__": self.num_moves >= self.episode_length}

        for i in range(self.N):
            self.attr[i] = max(0., min(1., self.attr[i] + self.impact * (r[i] - 1.)))
        tmp_alphas = np.array(self.alphas)
        for i in range(self.N):
            sumn = tmp_alphas[i] * max(0., 1. - self.attr[i] - self.threshold)
            self.alphas[i] -= sumn
            mean = np.array(self.attr) / self.attr.sum()
            cov = np.diag(mean) - np.dot(np.transpose([mean]), [mean])
            mig = np.random.multivariate_normal(sumn * mean, sumn * cov)
            for j in range(self.N):
                #self.alphas[i] += tmp_alphas[j] * max(0, 1 - self.attr[j] - self.threshold) * self.attr[i] / self.attr.sum()
                self.alphas[j] += mig[j]

        assert(abs(self.alphas.sum() - 1.) < eps)
        
        alphas = dict()
        for i in range(self.N):
            alphas[str(i)] = self.alphas[i] - tmp_alphas[i]
        
        info = dict()
        for i in range(self.N):
            info[str(i)] = {'policy': np.array(action[i]), 'reward': r[i], 'alphas': self.alphas[i]}

        return self.compute_states(), alphas, done, info

class BlockWithholdingEnv(MultiAgentEnv):
    def __init__(self, env_config):
        self.action_space = ACTION_SPACE
        self.observation_space = STATE_SPACE

        self.alphas = env_config['alphas']
        self.N = len(self.alphas)
        self.honest_power = 1 - sum(self.alphas)
        self.episode_length = env_config['ep_length']

        self.num_moves = 0
    
    #reset the environment to the starting state
    def reset(self):
        self.num_moves = 0
        return {
            '0': 0,
            '1': 0
        }
    
    def step(self, action_dict):
        self.num_moves += 1

        a = self.alphas[0]
        b = self.alphas[1]
        x = action_dict['0'][0]
        y = action_dict['1'][0]

        done = {"__all__": self.num_moves >= self.episode_length}

        R = compute_reward(a, b, x * a, y * b)
        info = dict()
        info['0'] = {'policy': x * a, 'reward': R['0']}
        info['1'] = {'policy': y * b, 'reward': R['1']}

        return {'0': 0, '1': 0}, R, done, info

class Constant(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.infiltrating = config['infiltrating']

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
            actions.append([self.infiltrating])
        return actions, [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

class NE_strategy(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        x, y, r1, r2 = get_Nash_equilibrium(config['alphas'])
        self.infiltrating = y / config['alphas'][1]
    
    
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
            actions.append([self.infiltrating])
        return actions, [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

def on_episode_start(info):
    episode = info["episode"]

def on_episode_step(info):
    episode = info["episode"]
    episode.user_data['0'] = episode.last_info_for('0')
    episode.user_data['1'] = episode.last_info_for('1')

def on_episode_end(info):
    episode = info["episode"]
    print(episode.user_data)

def run_RL(policies_to_train, policies):
    def select_policy(agent_id):
        return agent_id
    
    tune.run(
        args.algo,
        stop={"episodes_total": args.episodes},
        config={
            "num_gpus": args.gpus,
            "env": BlockWithholdingEnv,
            "entropy_coeff": 0.01,
            "entropy_coeff_schedule": args.episodes * 1000,
            "clip_param": 0.1,
            "gamma": args.gamma,
            "lambda": args.lmbda,
            "lr_schedule": [[0, 1e-5], [args.episodes, 1e-7]],
            "num_workers": args.workers,
            "num_envs_per_worker": 1,
            "sample_batch_size": 10,
            "train_batch_size": 128,
            "multiagent": {
                "policies_to_train": policies_to_train,
                "policies": policies,
                "policy_mapping_fn": select_policy,
            },
            "env_config": {
                "alphas":args.alphas,
                'ep_length':args.ep_length
            },
            "monitor": True,
            "callbacks": {
                "on_episode_start": on_episode_start,
                "on_episode_step": on_episode_step,
                "on_episode_end": on_episode_end,
            },
            "ignore_worker_failures": True,
        })


NE['a0'], NE['a1'], NE['r1'], NE['r2'] = get_Nash_equilibrium(args.alphas)
print(args.alphas, NE)

policies_to_train = [str(i) for i in range(len(args.alphas))]

policies = dict()
for i in range(len(args.alphas)):
    policies[str(i)] = (None, STATE_SPACE, ACTION_SPACE, {
        "model": {
            "use_lstm":args.use_lstm
        }
    })

run_RL(policies_to_train, policies)


