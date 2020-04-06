from gym import spaces

ADOPT = 0
OVERRIDE = 1
WAIT = 2
MATCH = 3
NORMAL = 0
FORKING = 1
CATCH_UP = 2

def make_spy_space(parties, blocks):
    state_space_list = list((spaces.Discrete(blocks + 3),
                    spaces.Discrete(blocks + 3),
                    spaces.Discrete(blocks + 3),
                    spaces.Discrete(3)))
    for i in range(2*parties + 1):
        state_space_list.append(spaces.Discrete(blocks + 3))
    return spaces.Tuple(state_space_list)

    