import mdptoolbox
import numpy as np
class OSM:
    def __init__(self, alpha, gamma, max_hidden_block):
        self._max_hidden_block = max_hidden_block
        self._alpha = alpha
        self._gamma = gamma
        self._state_vector_n = 3
        self._action_space_n = 4
        self._state_space = []
        
        # a < b
        for a in range(0, max_hidden_block + 1):
            for b in range(a + 1, max_hidden_block + 1):
                self._state_space.append((a, b, "normal"))
                self._state_space.append((a, b, "catch up"))
        #a = b = 0
        self._state_space.append((0, 0, "normal"))

        #a = b
        for a in range(1, max_hidden_block + 1):
            self._state_space.append((a, a, "normal"))
            self._state_space.append((a, a, "catch up"))
            self._state_space.append((a, a, "forking"))

        #a > b
        for a in range(1, max_hidden_block + 1):
            self._state_space.append((a, 0, "normal"))
            for b in range(1, a):
                self._state_space.append((a, b, "normal"))
                self._state_space.append((a, b, "forking"))

        #a = max_hidden_block + 1, b < a
        for b in range(0, max_hidden_block + 1):
            self._state_space.append((max_hidden_block + 1, b, "normal"))
            if (b > 0):
                self._state_space.append((max_hidden_block + 1, b, "forking"))

        #a < b, b = max_hidden_block + 1
        for a in range(0, max_hidden_block + 1):
            self._state_space.append((a, max_hidden_block + 1, "normal"))
        
        self._state_space_n = len(self._state_space)
        self._state_dict = dict(zip(self._state_space, range(0, self._state_space_n)))
        self.transition_matrix = np.zeros((self._action_space_n, self._state_space_n, self._state_space_n))
        self.reward_matrix = np.zeros((self._action_space_n, self._state_space_n, self._state_space_n))
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
        self.transition_matrix[a, s1_idx, s2_idx] = p
        self.reward_matrix[a, s1_idx, s2_idx] = r

    
    def MDP_matrix_init(self):
        self._matrix_init = True
        alpha = self._alpha
        gamma = self._gamma
        for action in range(self._action_space_n):
            for s1 in range(0, self._state_space_n):

                a, b, status = self._index_to_name(s1)

                legal = False

                # out of bound..force to override
                if (a == self._max_hidden_block + 1 and b < a):
                    #override, publish (b + 1) blocks
                    if (action == 1):
                        legal = True
                        self.add_transition(action, (a, b, status), (a - b - 1, 0, "normal"), 1, (b + 1) * (1 - alpha))

                # out of bound... force to give up
                elif (b == self._max_hidden_block + 1 and a < b):
                    #match -- abandon, accept b blocks
                    if (action == 0):
                        legal = True
                        self.add_transition(action, (a, b, status), (0, 0, "normal"), 1, b * (-alpha))
                elif (a < b):
                    # attacker abandons his private fork
                    if (action == 0):
                        legal = True
                        self.add_transition(action, (a, b, status), (0, 0, "normal"), 1, b * -alpha)
                    if (action == 2):
                        legal = True
                        if (a + 1 == b):
                            self.add_transition(action, (a, b, status), (a + 1, b, "catch up"), alpha, 0)
                        else:
                            self.add_transition(action, (a, b, status), (a + 1, b, "normal"), alpha, 0)

                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), 1 - alpha, 0)

                elif (a == b and a == 0):
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "normal"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), 1 - alpha, 0)

                elif (a == b and status == "normal"):
                    # attacker publishes all block and matches
                    if (action == 3):
                        legal = True
                        self.add_transition(action, (a, b, status), (a, b, "forking"), 1, 0)
                    # wait
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "normal"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), 1 - alpha, 0)

                elif (a == b and status == "catch up"):
                    # in this situation, the attacker cannot match!
                    # wait
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "normal"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), 1 - alpha, 0)

                elif (a == b and status == "forking"):
                    # wait, 3 fork possibilities
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "forking"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a - b, 1, "normal"), (1 - alpha) * gamma, b * (1-alpha))
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), (1 - alpha) * (1 - gamma), 0)

                elif (a > b and b == 0):
                    # override, publish a block
                    if (action == 1):
                        legal = True
                        self.add_transition(action, (a, b, status), (a - 1, 0, "normal"), 1, 1-alpha)
                    # wait
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "normal"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), 1 - alpha, 0)

                elif (a > b and b > 0 and status == "normal"):
                    # match, publish b blocks
                    if (action == 3):
                        legal = True
                        self.add_transition(action, (a, b, status), (a, b, "forking"), 1, 0)
                    # override, publish (b + 1) blocks
                    if (action == 1):
                        legal = True
                        self.add_transition(action, (a, b, status), (a - b - 1, 0, "normal"), 1, (b + 1) * (1-alpha))
                    # wait
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "normal"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), 1 - alpha, 0)

                elif (a > b and b > 0 and status == "forking"):
                    # don't need match...
                    # override, publish (b + 1) blocks
                    if (action == 1):
                        legal = True
                        self.add_transition(action, (a, b, status), (a - b - 1, 0, "normal"), 1, (b + 1) * (1 - alpha))
                    # wait, 3 fork possibilities
                    if (action == 2):
                        legal = True
                        self.add_transition(action, (a, b, status), (a + 1, b, "forking"), alpha, 0)
                        self.add_transition(action, (a, b, status), (a - b, 1, "normal"), (1 - alpha) * gamma, b * (1 - alpha))
                        self.add_transition(action, (a, b, status), (a, b + 1, "normal"), (1 - alpha) * (1 - gamma), 0)

                if (legal == False):
                    self.add_transition(action, (a, b, status), (a, b, status), 1, -1000000)

        mdptoolbox.util.check(self.transition_matrix, self.reward_matrix)
    def get_MDP_matrix(self):
        if (self._matrix_init == False): self.MDP_matrix_init()
        return self.transition_matrix, self.reward_matrix