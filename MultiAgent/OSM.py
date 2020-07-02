import mdptoolbox
import numpy as np
import sys
class OSM:
    def __init__(self, alpha, gamma, max_hidden_block):
        self._max_hidden_block = max_hidden_block
        self._alpha = alpha
        self._gamma = gamma
        self._state_vector_n = 3
        self._action_space_n = 4
        self._state_space = []
        
        for a in range(max_hidden_block + 2):
            for b in range(max_hidden_block + 2):
                if a >= b:
                    self._state_space.append((a,b,"forking"))
                self._state_space.append((a,b,"catch up"))
                self._state_space.append((a,b,"normal"))
        
        self._state_space_n = len(self._state_space)
        self._state_dict = dict(zip(self._state_space, range(0, self._state_space_n)))
        self.transition_matrix = np.zeros((self._action_space_n, self._state_space_n, self._state_space_n))
        self.reward_matrix = np.zeros((self._action_space_n, self._state_space_n, self._state_space_n))
        print('MDP transition matrix size',self.transition_matrix.shape)
    @property
    def observation_space_n(self):
        return self._state_space_n
    def _name_to_index(self, s):
        return self._state_dict[s]
    #input a state index, return its description
    def _index_to_name(self, idx):
        return self._state_space[idx]
    # add a transition to MDP matrices
    def add_transition(self, a, s1, s2, p, r):
        
        s1_idx = self._name_to_index(s1)
        s2_idx = self._name_to_index(s2)
        self.transition_matrix[a, s1_idx, s2_idx] += p
        self.reward_matrix[a, s1_idx, s2_idx] += r

    
    def MDP_matrix_init(self):
        np.set_printoptions(threshold=sys.maxsize)
        self._matrix_init = True
        alpha = self._alpha
        gamma = self._gamma
        for action in range(self._action_space_n):
            for s1 in range(0, self._state_space_n):

                a, b, status = self._index_to_name(s1)

                legal = False

                # out of bound..force to override
                if a == self._max_hidden_block + 1 and b < a and action == 1:
                    legal = True
                    self.add_transition(action, (a, b, status), (a - b, 0, "catch up"), alpha, (b+1)*(1-alpha))
                    self.add_transition(action, (a, b, status), (a - b - 1, 1, "normal"), 1 - alpha, (b + 1) * (1 - alpha))
                # out of bound... force to give up
                elif b == self._max_hidden_block + 1 and a <= b and action == 0:
                    #match -- abandon, accept b blocks
                    legal = True
                    self.add_transition(action, (a, b, status), (1, 0, "catch up"), alpha, b * (-alpha))
                    self.add_transition(action, (a, b, status), (0, 1, "catch up"), 1 - alpha, b * (-alpha))
                elif action == 0 and b <= self._max_hidden_block and a <= self._max_hidden_block:
                    legal = True
                    self.add_transition(action, (a, b, status), (1, 0, "catch up"), alpha, b * (-alpha))
                    self.add_transition(action, (a, b, status), (0, 1, "catch up"), 1 - alpha, b * (-alpha))
                    

                elif action == 1 and a > b and b <= self._max_hidden_block and a <= self._max_hidden_block:
                    legal = True
                    self.add_transition(action, (a, b, status), (a - b, 0, "catch up"), alpha, (b+1)*(1-alpha))
                    self.add_transition(action, (a, b, status), (a - b - 1, 1, "normal"), 1 - alpha, (b + 1) * (1 - alpha)) 
                elif action == 2 and (status == "catch up" or status == "normal") and b <= self._max_hidden_block and a <= self._max_hidden_block:
                    legal = True
                    self.add_transition(action, (a,b,status), (a + 1, b, "catch up"), alpha, 0)
                    self.add_transition(action, (a,b,status), (a, b + 1, "normal"), 1 - alpha, 0)
                elif action == 2 and status == "forking" and b <= self._max_hidden_block and a <= self._max_hidden_block:
                    self.add_transition(action, (a,b,status), (a+1, b, "forking"), alpha, 0)
                    self.add_transition(action, (a,b,status), (a - b, 1, "normal"), gamma *(1-alpha), alpha*b)
                    self.add_transition(action, (a,b,status), (a, b+1, "normal"), (1-gamma)*(1-alpha), 0)
                elif action == 3 and status == "normal" and a >= b and b > 0 and b <= self._max_hidden_block and a <= self._max_hidden_block:
                    self.add_transition(action, (a,b,status), (a+1, b, "forking"), alpha, 0)
                    self.add_transition(action, (a,b,status), (a - b, 1, "normal"), gamma *(1-alpha), alpha*b)
                    self.add_transition(action, (a,b,status), (a, b+1, "normal"), (1-gamma)*(1-alpha), 0)
                else:
                    self.add_transition(action, (a,b,status), (a,b,status), 1, -100000)
        mdptoolbox.util.check(self.transition_matrix, self.reward_matrix)
    def get_MDP_matrix(self):
        if (self._matrix_init == False): self.MDP_matrix_init()
        return self.transition_matrix, self.reward_matrix