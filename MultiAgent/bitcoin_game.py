import random
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import argparse
import ray
from gym import spaces
from ray.tune.registry import register_env
from ray.rllib.models.preprocessors import get_preprocessor
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer
#from ray.rllib.agents.pg.pg import PGTrainer
#from ray.rllib.agents.pg.pg_policy import PGTFPolicy
from ray.rllib.policy.policy import Policy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils import try_import_tf
#from ray.tune.util import flatten_dict
from ray.tune.result import (NODE_IP, TRAINING_ITERATION, TIME_TOTAL_S,
                             TIMESTEPS_TOTAL, EXPR_PARAM_FILE,
                             EXPR_PARAM_PICKLE_FILE, EXPR_PROGRESS_FILE,
                             EXPR_RESULT_FILE)
from functools import reduce
from itertools import (chain, takewhile)
from ray.rllib.agents.ppo import PPOTrainer             
#from OSM import OSM
import csv
import math
import time
import constants
import mdptoolbox
from OSM import OSM
from BitcoinEnv import BitcoinEnv
from ray.rllib.models import ModelCatalog
from ParametricActionsModel import ParametricActionsModel
from ParametricBitcoin import ParametricBitcoin

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

def on_train_result(info):
    result = info["result"]
    episodes_total = result['episodes_total']
    learner_stats = result['info']['learner']
    trainer = info["trainer"]
    trainer.workers.foreach_worker(
        lambda ev: ev.foreach_env(
            lambda env: env.wrapped.set_phase(episodes_total, learner_stats)))

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
            if curr_s[0] > curr_s[1]:
                return 1
            else:
                return 0
    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        actions = []
        for obs in obs_batch:
            a = int(round(obs[0]))
            h = int(round(obs[1]))
            o = int(round(obs[2]))
            f = int(round(obs[3]))
            actions.append(self.OSM_act([a,h,o,f]))
        return actions, [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass
class Honest(Policy):
    def __init__(self, observation_space, action_space, config):
        Policy.__init__(self, observation_space, action_space, config)
        self.blocks = config['blocks']
        self.fiftyone = config['fiftyone']
        self.extended = config['extended']
    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        actions = []
        for obs in obs_batch:
            a = int(round(obs[0]))
            h = int(round(obs[1]))
            o = int(round(obs[2]))
            f = int(round(obs[3]))
            if self.extended:
                if a > h:
                    actions.append(5)
                else:
                    actions.append(0)
            elif a >= self.blocks or h >= self.blocks:
                if a > h:
                    actions.append(1)
                else:
                    actions.append(0)
            elif self.fiftyone:
                listobs = list(obs)
                maxlen = max(listobs[4:-1])
                if a > maxlen:
                    actions.append(5)
                else:
                    actions.append(4)
            else:
                if a > h:
                    actions.append(1)
                else:
                    actions.append(0)
        return actions, [], {}

    def learn_on_batch(self, samples):
        pass

    def get_weights(self):
        pass

    def set_weights(self, weights):
        pass

# runs a fixed strategy
def run_saved(args):
    if args.OSM[0] == 1 and args.OSM[1] == 0:
        setting = "RLvsOSM"
    elif args.OSM[0] == 1 and args.OSM[1] == 1:
        setting = "OSMvsOSM"
    else:
        setting = "RL{0}".format(len(args.alphas) - sum(args.honest))
    if args.save_path == 'none':
        checkpointnum = 0
    else:
        checkpointnum = args.save_path.split('-')[-1]
    env_name = "{setting}_{spirit}_{blocks}_{alpha:04d}_{spy}_{checkpointnum}".format(spirit = int(args.team_spirit*100), 
                blocks = int(args.blocks), 
                alpha = int(args.alphas[0]*10000), 
                spy = args.spy[1],
                setting = setting,
                checkpointnum = checkpointnum)
    ray.init(local_mode=True, memory=700 * 1024 * 1024, object_store_memory=100 * 1024 * 1024, driver_object_store_memory=100 * 1024 * 102)
    print("Testing {0}".format(setting), env_name)
    def select_policy(agent_id):
        return agent_id
    ModelCatalog.register_custom_model(
        "pa_model", ParametricActionsModel)
    register_env(env_name, lambda config: ParametricBitcoin(config))
    
    if args.extended:
        action_n = 6
    else:
        action_n = 4
    # define the state space, one for parties that have access to spy info and one without
    spy_state_space = constants.make_spy_space(len(args.alphas), args.blocks)
    blind_state_space = constants.make_blind_space(len(args.alphas), args.blocks)
    policies = dict()
    osm_space = spaces.Box(low=np.zeros(4), 
                high=np.array([args.blocks + 4, args.blocks + 4, args.blocks + 4, 3.]))
    if sum(args.OSM) > 0:
        osm = OSM_strategy(osm_space,spaces.Discrete(4), {'alpha': args.alphas[0], 'gamma': args.gammas[0], 'blocks': args.blocks})
    
    blind_dim = 0
    for space in blind_state_space:
        blind_dim +=  get_preprocessor(space)(space).size 
        
    spy_dim = 0
    for space in spy_state_space:
        spy_dim += get_preprocessor(space)(space).size
    
    spy_state_space_wrapped = spaces.Dict(
        {   "action_mask": spaces.Box(0,1,shape = (action_n,), dtype=np.int64),
            "avail_actions": spaces.Box(-10, 10, shape=(action_n, action_n), dtype=np.float64),
            "bitcoin": spaces.Box(0,np.inf,shape=(spy_dim,))
        } 
    )
    blind_state_space_wrapped = spaces.Dict(
        {   "action_mask": spaces.Box(0,1,shape = (action_n,), dtype=np.int64),
            "avail_actions": spaces.Box(-10, 10, shape=(action_n, action_n), dtype=np.float64),
            "bitcoin": spaces.Box(0,np.inf,shape=(blind_dim,), dtype=np.float64)
        } 
    )
    preps = [None for i in range(len(args.alphas))]
    for i in range(len(args.alphas)):
        if args.spy[i] == 1:
            policies[str(i)] = (None, spy_state_space_wrapped, spaces.Discrete(action_n), {
                        "model": {
                            "use_lstm":args.use_lstm,
                            "custom_model": "pa_model",
                            "custom_options": {
                                "parties": len(args.alphas),
                                "spy": True,
                                "blocks": args.blocks,
                                "extended": args.extended
                            }
                        }
                    })
            preps[i] = get_preprocessor(spy_state_space_wrapped)(spy_state_space_wrapped)
        elif args.OSM[i] == 1:
            policies[str(i)] = (OSM_strategy, osm_space, spaces.Discrete(4), {'alpha': args.alphas[0], 'gamma': args.gammas[0], 'blocks': args.blocks})
        elif args.honest[i] == 1:
            policies[str(i)] = (Honest, osm_space, spaces.Discrete(6), {'alpha': args.alphas[0], 'gamma': args.gammas[0], 'blocks': args.blocks, 'fiftyone': args.fiftyone[i], 'extended': args.extended})
        else:
            policies[str(i)] = (None, blind_state_space_wrapped, spaces.Discrete(action_n), {
                            "model": {
                                "use_lstm":args.use_lstm,
                                "custom_model": "pa_model",
                                "custom_options": {
                                "parties": len(args.alphas),
                                "spy": False,
                                "blocks": args.blocks,
                                "extended": args.extended
                            }
                        }
                    })
            preps[i] = get_preprocessor(blind_state_space_wrapped)(blind_state_space_wrapped)
    env_config = {'max_hidden_block': args.blocks,
            'alphas': args.alphas,
            'gammas': args.gammas,
            'ep_length': args.ep_length,
            'print': args.debug,
            'spy': args.spy,
            'team_spirit': args.team_spirit,
            'OSM': args.OSM,
            'extended': args.extended,
            'honest': args.honest,
        }
    policies_to_train = [str(i) for i in range(len(args.alphas)) if args.OSM[i] != 1 and args.honest[i] != 1]
    env = ParametricBitcoin(env_config=env_config)
    if len(policies_to_train) != 0:
        if args.trainer == 'PPO':
            trainer = PPOTrainer(env=BitcoinEnv, config={
                        "num_workers": 0,
                        "multiagent": {
                            "policies_to_train": policies_to_train,
                            "policies": policies,
                            "policy_mapping_fn": select_policy,
                        },
                        "env_config": env_config
                    })
        else:
            trainer = DQNTrainer(env=env_name, config={
                        "eager": True,
                        "multiagent": {
                            "policies_to_train": policies_to_train,
                            "policies": policies,
                            "policy_mapping_fn": select_policy,
                        },
                        "env_config": env_config
                    })
            model = trainer.get_policy().model
            print(model.base_model.summary())
        print("Restoring model")
        trainer.restore(args.save_path) 
    loaded_policies = dict()
    for k in range(len(args.alphas)):
        if args.OSM[k] == 1:
            loaded_policies[str(k)] = osm
        elif args.honest[k] == 1:
            honest = Honest(osm_space, spaces.Discrete(6), {'alpha': args.alphas[0], 'gamma': args.gammas[0], 'blocks': args.blocks, 'fiftyone': args.fiftyone[k], 'extended': args.extended}, )
            loaded_policies[str(k)] = honest
            preps[k] = None
        else:
            loaded_policies[str(k)] = trainer.get_policy(str(k))
    trials = 100000
    reslist = []
    for j in range(3):
        blocks = np.zeros(len(args.alphas) + 1)
        event_blocks = np.zeros(len(args.alphas) + 1)
        action_dist = {str(i): np.zeros(action_n) for i in range(len(args.alphas))}
        res = dict()
        for i in range(trials):
            obs = env.reset()
            isDone = False
            RNNstates = {str(i): [] for i in range(len(args.alphas))}
            while not isDone:
                action_dict = dict()
                for k in range(len(policies)):
                    prep = preps[k]
                    if not prep:
                        action_dict[str(k)], _, _ = loaded_policies[str(k)].compute_single_action(obs=obs[str(k)], state = [])
                    else:   
                        action_dict[str(k)], _, _ = loaded_policies[str(k)].compute_single_action(obs=prep.transform(obs[str(k)]), state = [])
                    action_dist[str(k)][action_dict[str(k)]] += 1
                obs, _, done, _ = env.step(action_dict)
                isDone = done['__all__']
            if i == 0 and j == 0:
                with open(os.path.join('/afs/ece/usr/charlieh/eval_results', env_name + '_trace.txt'), 'w+') as f:
                    f.write(env.wrapped._debug_string)
            blocks += env.wrapped._accepted_blocks
            event_blocks += env.wrapped._total_blocks
            total_event_blocks = np.sum(event_blocks)
            if i % 100 == 0:
                print("Relative rewards", blocks/np.sum(blocks))
                print("Relative received", event_blocks/total_event_blocks)
                for i in range(len(args.alphas)):
                    print("Action dist", str(i), action_dist[str(i)]/np.sum(action_dist[str(i)]))
        res['blocks'] = blocks
        res['action dist'] = action_dist
        res['blocks norm'] = blocks/np.sum(blocks)
        res['actions norm'] = {str(i):action_dist[str(i)]/np.sum(action_dist[str(i)]) for i in range(len(args.alphas))}
        reslist.append(res)
    np.save(os.path.join('/afs/ece/usr/charlieh/eval_results', env_name), reslist, allow_pickle=True)

def run_RL(args):
    if args.save_path == 'osmvsosm':
        env_name = "OSMvsOSM_{spirit:03d}_{blocks}_{alpha:04d}_{spy}".format(spirit = int(args.team_spirit*100), 
                blocks = int(args.blocks), 
                alpha = int(args.alphas[0]*10000), 
                spy = args.spy[1],
                players = len(args.alphas))
    elif args.OSM[0] == 1:
        env_name = "RLvsOSM_{spirit:03d}_{blocks}_{alpha:04d}_{spy}".format(spirit = int(args.team_spirit*100), 
                blocks = int(args.blocks), 
                alpha = int(args.alphas[0]*10000), 
                spy = args.spy[1],
                players = len(args.alphas))
    else:
        env_name = "RL{players}_{spirit:03d}_{blocks}_{alpha:04d}_{spy}".format(spirit = int(args.team_spirit*100), 
                    blocks = int(args.blocks), 
                    alpha = int(args.alphas[0]*10000), 
                    spy = args.spy[0],
                    players = len(args.alphas) - sum(args.honest))
    ray.init(temp_dir=os.path.join('/tmp/', env_name) + time.strftime("%Y%m%d-%H%M%S"))
    def select_policy(agent_id):
        return agent_id
    mine_act_space = spaces.Discrete(6)
    ModelCatalog.register_custom_model(
        "pa_model", ParametricActionsModel)
    register_env(env_name, lambda config: ParametricBitcoin(config))
    if args.trainer == "DQN":
        cfg = {
            # TODO(ekl) we need to set these to prevent the masked values
            # from being further processed in DistributionalQModel, which
            # would mess up the masking. It is possible to support these if we
            # defined a a custom DistributionalQModel that is aware of masking.
            "hiddens": [],
            "dueling": False,
            "target_network_update_freq": 100*1000,
            "train_batch_size": 2**14,
            "buffer_size": 100000,
            "lr_schedule": [[0, 5e-4], [args.episodes*100/2, 1e-6]],
            "n_step": 10,
            "double_q": True,
        }
    elif args.trainer == "A2C":
        cfg = {
            "microbatch_size": 2**11,
            "train_batch_size": 2**11,
            "lr":3e-4,
            "rollout_fragment_length": 200,
        }
    else:
        if args.save_path == 'none':
            cfg = {
                #"use_critic": False,
                #"use_gae": False,
                "train_batch_size": 2**20,
                "sgd_minibatch_size": 2**14,
                "num_sgd_iter":2**6,
                #"kl_coeff": 10e-14,
                #"clip_param": 1000,
                #"train_batch_size": 2**18,
                #"sgd_minibatch_size": 2**18,
                #"entropy_coeff": .005,
                "rollout_fragment_length": (99*(3 + 1) + 1),
                "entropy_coeff_schedule": [[0,0.01],[args.episodes*100*4/2,0]],
                #"lr_schedule": [[0, 5e-3], [args.episodes*100*(len(args.alphas) + 1)/2, 5e-6]],
                "lr": 5e-4
            }
        else:
            cfg = {
                #"use_critic": False,
                #"use_gae": False,
                "train_batch_size": 397e3,
                "sgd_minibatch_size": 1,
                "num_sgd_iter":1,
                #"kl_coeff": 10e-14,
                #"clip_param": 1000,
                #"train_batch_size": 2**18,
                #"sgd_minibatch_size": 2**18,
                #"entropy_coeff": .005,
                #"rollout_fragment_length": (99*(3 + 1) + 1),
                #"entropy_coeff_schedule": [[0,0.01],[args.episodes*100*4/2,0]],
                #"lr_schedule": [[0, 5e-3], [args.episodes*100*(len(args.alphas) + 1)/2, 5e-6]],
                "lr": 1e-7
            }
    if args.extended:
        action_n = 6
    else:
        action_n = 4
    # define the state space, one for parties that have access to spy info and one without
    spy_state_space = constants.make_spy_space(len(args.alphas), args.blocks)
    blind_state_space = constants.make_blind_space(len(args.alphas), args.blocks)
    policies = dict()
    osm_space = spaces.Box(low=np.zeros(4), 
                high=np.array([args.blocks + 4, args.blocks + 4, args.blocks + 4, 3.]), dtype=np.float64)
    blind_dim = 0
    for space in blind_state_space:
        blind_dim +=  get_preprocessor(space)(space).size 
        
    spy_dim = 0
    for space in spy_state_space:
        spy_dim += get_preprocessor(space)(space).size
    
    spy_state_space_wrapped = spaces.Dict(
        {   "action_mask": spaces.Box(0,1,shape = (action_n,), dtype=np.int64),
            "avail_actions": spaces.Box(-10, 10, shape=(action_n, action_n), dtype=np.float64),
            "bitcoin": spaces.Box(0,np.inf,shape=(spy_dim,))
        } 
    )
    blind_state_space_wrapped = spaces.Dict(
        {   "action_mask": spaces.Box(0,1,shape = (action_n,), dtype=np.int64),
            "avail_actions": spaces.Box(-10, 10, shape=(action_n, action_n), dtype=np.float64),
            "bitcoin": spaces.Box(0,np.inf,shape=(blind_dim,), dtype=np.float64)
        } 
    )
    for i in range(len(args.alphas)):
        if args.spy[i] == 1 and args.OSM[i] == 0:
            policies[str(i)] = (None, spy_state_space_wrapped, spaces.Discrete(action_n), {
                        "model": {
                            "use_lstm":args.use_lstm,
                            "custom_model": "pa_model",
                            "custom_options": {
                                "parties": len(args.alphas),
                                "spy": True,
                                "blocks": args.blocks,
                                "extended": args.extended
                            }
                        }
                    })
        elif args.OSM[i] == 1:
            policies[str(i)] = (OSM_strategy, osm_space, spaces.Discrete(4), {'alpha': args.alphas[0], 'gamma': args.gammas[0], 'blocks': args.blocks})
        elif args.honest[i] == 1:
            policies[str(i)] = (Honest, osm_space, spaces.Discrete(6), {'alpha': args.alphas[0], 'gamma': args.gammas[0], 'blocks': args.blocks, 'fiftyone': args.fiftyone[i], 'extended': args.extended})
        else:
            policies[str(i)] = (None, blind_state_space_wrapped, spaces.Discrete(action_n), {
                            "model": {
                                "use_lstm":args.use_lstm,
                                "custom_model": "pa_model",
                                "custom_options": {
                                "parties": len(args.alphas),
                                "spy": False,
                                "blocks": args.blocks,
                                "extended": args.extended
                            }
                        }
                    })
    print(args)
    print(policies)
    if args.save_path == 'none':
        env_config = {'max_hidden_block': args.blocks,
                'alphas': args.alphas,
                'gammas': args.gammas,
                'ep_length': args.ep_length,
                'print': args.debug,
                'spy': args.spy,
                'team_spirit': args.team_spirit,
                'OSM': args.OSM,
                'extended': args.extended,
                'honest': args.honest,
                'initial_OSM': True,
                'wait_bias': True,
                'anneal_delay': True
            }
    else:
        env_config = {'max_hidden_block': args.blocks,
                'alphas': args.alphas,
                'gammas': args.gammas,
                'ep_length': args.ep_length,
                'print': args.debug,
                'spy': args.spy,
                'team_spirit': args.team_spirit,
                'OSM': args.OSM,
                'extended': args.extended,
                'honest': args.honest,
            }
    if args.save_path == 'none':
        config=dict({
                "env": env_name,
                #"vf_share_layers": True,
                "gamma": 0.997,
                "num_workers": args.workers,
                "batch_mode": "complete_episodes",
                "multiagent": {
                    "policies_to_train": [str(i) for i in range(len(args.alphas)) if args.OSM[i] != 1 and args.honest[i] != 1],
                    "policies": policies,
                    "policy_mapping_fn": select_policy,
                },
                "log_level": 'ERROR',
                "env_config": env_config,
                "num_gpus": 0,
                "callbacks": {
                    "on_episode_start": on_episode_start,
                    "on_episode_step": on_episode_step,
                    "on_episode_end": on_episode_end,
                    "on_train_result": on_train_result
                }
            
            }, **cfg)
    else:
        config=dict({
                "env": env_name,
                #"vf_share_layers": True,
                "gamma": 0.997,
                "num_workers": args.workers,
                "batch_mode": "complete_episodes",
                "multiagent": {
                    "policies_to_train": [str(i) for i in range(len(args.alphas)) if args.OSM[i] != 1 and args.honest[i] != 1],
                    "policies": policies,
                    "policy_mapping_fn": select_policy,
                },
                "log_level": 'ERROR',
                "env_config": env_config,
                "num_gpus": 0,
                "callbacks": {
                    "on_episode_start": on_episode_start,
                    "on_episode_step": on_episode_step,
                    "on_episode_end": on_episode_end,
                    "on_train_result": on_train_result
                }
            
            }, **cfg)
    if args.save_path != 'none':
        if args.save_path == 'osmvsosm':
            tune.run(
                args.trainer,
                name='eval_v3',
                stop={"iterations_since_restore": 300},
                checkpoint_score_attr='episode_reward_mean',
                config=config,
                local_dir = "/afs/ece/usr/charlieh/ray_results",
                checkpoint_at_end=True
            ) 
        else:
            tune.run(
                args.trainer,
                name='eval_v3',
                stop={"iterations_since_restore": 300},
                restore=args.save_path,
                checkpoint_score_attr='episode_reward_mean',
                config=config,
                local_dir = "/afs/ece/usr/charlieh/ray_results",
                #checkpoint_freq=20,
                #keep_checkpoints_num=1,
                checkpoint_at_end=True
            ) 
    else:
        if hasattr(args, 'cont_path'):
            tune.run(
                args.trainer,
                name='RL4_v4',
                stop={"episodes_total": args.episodes},
                restore=args.cont_path,
                checkpoint_score_attr='episode_reward_mean',
                config=config,
                local_dir = "/afs/ece/usr/charlieh/ray_results",
                checkpoint_freq=1,
                #keep_checkpoints_num=1,
                checkpoint_at_end=True
            ) 
        else:
            tune.run(
                args.trainer,
                name='RL5',
                stop={"episodes_total": args.episodes},
                checkpoint_score_attr='episode_reward_mean',
                config=config,
                local_dir = "/afs/ece/usr/charlieh/ray_results",
                checkpoint_freq=1,
                #keep_checkpoints_num=1,
                checkpoint_at_end=True
            ) 
   
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
    CLI.add_argument('--episodes', type = int, default = 2*2*532500)
    #CLI.add_argument('--episodes', type = int, default = 20000)
    CLI.add_argument('--ep_length', type = int, default = 100)
    # give particular players the ability to see the hidden states of other
    # players; 1 means we have spy mining, 0 means no spy mining for a player
    CLI.add_argument('--spy', nargs='*', type=int, default = [0, 0])
    # number of OSM players
    CLI.add_argument('--OSM', type = int, default = 0)
    CLI.add_argument('--workers', type = int, default = min(len(os.sched_getaffinity(0)) - 1, 31))
    # the number of strategic players (i.e. not honest)
    CLI.add_argument('--players', type = int, default = 0)
    # how much to value the team (all non-honest miners)
    # (1 is fully support team, 0 is lone wolf.)
    CLI.add_argument('--team_spirit', type = float,default = 0.)
    # print the blockchain graphs at each step to see what's going on
    CLI.add_argument('--debug', type = bool, default = False)
    # input the global path of the checkpoint you want to run an evaluation on
    CLI.add_argument('--save_path', type = str, default = '')
    CLI.add_argument('--use_lstm', type = str, default = False)
    CLI.add_argument('--extended', type=int, default = 0)
    CLI.add_argument('--honest', type=int, default = [0,1])
    args = CLI.parse_args()  
    
    if args.players > 0:
        alpha = args.alphas[0]
        spy = args.spy[0]
        args.alphas = [alpha]*args.players
        args.gammas = [1./args.players]*args.players
        args.spy = [spy]*args.players
        args.honest = [0]*args.players
        args.OSM = [1]*args.OSM + [0]*(args.players - args.OSM)
        
        
        args.alphas = args.alphas + [1 - alpha*args.players]
        args.gammas = args.gammas + [0]
        args.spy = args.spy + [0]
        args.honest = args.honest + [1]
        args.OSM = args.OSM + [0]
        args.fiftyone = [0]*(args.players + 1)
    if sum(args.OSM) == 2:
        args.alphas[2] = 0
        args.alphas[3] = 1 - sum(args.alphas[:2])
    args.cont_path = args.save_path
    args.save_path = 'none'
    run_RL(args)
    
    
if __name__ == "__main__":
    main()