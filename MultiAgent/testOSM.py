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
import math
import time
import constants
import mdptoolbox

from OSM import OSM
from BitcoinEnv import BitcoinEnv
from bitcoin_game import OSM_strategy
'''
blocks = 5
osm_space = spaces.Box(low=np.zeros(4), 
                high=np.array([blocks + 4, blocks + 4, blocks + 4, 3.]))
osm = OSM_strategy(osm_space, spaces.Discrete(4), {'alpha':.15, 'gamma':0,'blocks':5})
print(osm.OSM_act([1, 1, 1, 0]))
'''

osm = OSM(.15,.5,5)
osm.MDP_matrix_init()
P, R = osm.get_MDP_matrix()
solver = mdptoolbox.mdp.PolicyIteration(P, R, 0.99)
solver.run()
print(solver.V)
