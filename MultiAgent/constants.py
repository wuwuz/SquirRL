from gym import spaces
import numpy as np
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
                    
    