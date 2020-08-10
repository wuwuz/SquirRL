from gym import spaces
import numpy as np
from ray.rllib.policy.policy import Policy
from OSM import OSM
import mdptoolbox

ADOPT = 0
OVERRIDE = 1
WAIT = 2
MATCH = 3
NORMAL = 0
FORKING = 1
CATCH_UP = 2

def make_spy_space(parties, blocks):
    return spaces.Tuple((spaces.Box(low=np.array([0.]*(parties + 4)), high=np.array([blocks+3]*(parties + 3) + [np.inf]), 
        shape=(parties + 4,)), spaces.Discrete(3)))
def make_blind_space(parties, blocks):
    return spaces.Tuple((spaces.Box(low = np.array([0.]*4), high=np.array([blocks + 3]*3 + [np.inf]), shape=(4,)),
                    spaces.Discrete(3)))
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
        self.extended = config['extended']
    def OSM_act(self, s):
        curr_s = list(s)
        if s[3] == NORMAL:
            curr_s[3] = 'normal'
        elif s[3] == FORKING:
            curr_s[3] = 'forking'
        else:
            curr_s[3] = 'catch up'
        smaller_state = curr_s[:2] + [curr_s[3]]
        smaller_state = tuple(smaller_state)
        if curr_s[0] >= self.blocks or curr_s[1] >= self.blocks:
            if curr_s[0] > curr_s[1]:
                if self.extended:
                    return 5
                else:
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
    