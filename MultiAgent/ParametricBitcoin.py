import gym
from gym.spaces import Box, Dict, Discrete
import numpy as np 
from BitcoinEnv import BitcoinEnv
from ray.rllib.models.preprocessors import get_preprocessor
import constants
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from collections import deque
import tree_draw_fcns
from functools import reduce
from itertools import (chain, takewhile)
class ParametricBitcoin(MultiAgentEnv):
    def __init__(self, env_config):
        if env_config['extended']:
            self.action_n = 6
        else:
            self.action_n = 4
        
        self.extended = env_config['extended']
        self.action_space = Discrete(self.action_n)
        self.wrapped = BitcoinEnv(env_config)
        self.config = env_config
        self.alphas = env_config['alphas']
        self.max_hidden_block = env_config['max_hidden_block']
        self.game_trace = deque(''*10, 10)
        self.observation_space = Dict({
            "action_mask": Box(0,1,shape = (self.action_n,)),
            "avail_actions": Box(-10, 10, shape=(self.action_n, self.action_n)),
            "bitcoin":self.wrapped.observation_space,
        })
        spy_space = constants.make_spy_space(len(self.alphas), self.max_hidden_block)
        blind_space = constants.make_blind_space(len(self.alphas), self.max_hidden_block)
        self.prep = get_preprocessor(Discrete(3))(Discrete(3))
        self.action_assignments = np.zeros((self.action_n, self.action_n))
        for i in range(self.action_n):
            self.action_assignments[i,i] = 1
    def update_avail_actions(self, obs):
        self.action_masks = [np.array([1]*self.action_n) for i in range(len(self.alphas))]
        for agent,currobs in obs.items():
            i = int(agent)
            a = currobs[0]
            b = currobs[1]
            status = currobs[3]
            if status == 0:
                status = "normal"
            elif status == 1:
                status = "forking"
            elif status == 2:
                status = "catch up"
            if a >= self.max_hidden_block or b >= self.max_hidden_block:
                if a > b:
                    self.action_masks[i][0] = 0
                    
                    self.action_masks[i][2] = 0
                    self.action_masks[i][3] = 0
                    if self.extended:
                        self.action_masks[i][1] = 0
                        self.action_masks[i][4] = 0
                elif a <= b:
                    self.action_masks[i][1] = 0
                    self.action_masks[i][2] = 0
                    self.action_masks[i][3] = 0
                    if self.extended:
                        self.action_masks[i][4] = 0
                        self.action_masks[i][5] = 0
            elif a < b:
                self.action_masks[i][1] = 0
                self.action_masks[i][3] = 0
            elif a == b and a == 0:
                self.action_masks[i][1] = 0
                self.action_masks[i][3] = 0
            elif a == b and status == "normal":
                self.action_masks[i][1] = 0

            elif a == b and (status == "catch up" or status == "forking"):
                self.action_masks[i][1] = 0
                self.action_masks[i][3] = 0
            elif a > b and b == 0:
                self.action_masks[i][0] = 0
                self.action_masks[i][3] = 0
                if self.extended:
                    self.action_masks[i][4] = 0
            elif a > b and b > 0 and status == "normal":
                self.action_masks[i][0] = 0
                if self.extended:
                    self.action_masks[i][4] = 0
            elif a > b and b > 0 and (status == "forking" or status == "catch up"):
                self.action_masks[i][0] = 0
                self.action_masks[i][3] = 0
                if self.extended:
                    self.action_masks[i][4] = 0
                
    def obsdict(self, orig_obs):
        returned_dict = dict()
        for i in range(len(self.alphas)):
            origobs = list(orig_obs[str(i)])
            try:
                if self.config['OSM'][i] == 1 or self.config['honest'][i] == 1:
                    returned_dict[str(i)] = np.array(orig_obs[str(i)][:4])
                elif self.config['spy'][i] == 1:
                    box_component = np.array(origobs[:2] + origobs[4:])
                    discrete_component = self.prep.transform(origobs[3])
                    obstuple = np.concatenate([box_component, discrete_component], axis = 0)
                    returned_dict[str(i)] = {
                        "action_mask": self.action_masks[i],
                        "avail_actions":self.action_assignments,
                        "bitcoin": np.array(obstuple).flatten()
                    }
                else:                    
                    box_component = np.array(origobs[:2] + origobs[4:])
                    discrete_component = self.prep.transform(origobs[3])
                    obstuple = np.concatenate([box_component, discrete_component], axis = 0)
                    returned_dict[str(i)] = {
                        "action_mask": self.action_masks[i],
                        "avail_actions":self.action_assignments,
                        "bitcoin": np.array(obstuple).flatten()
                        # "bitcoin": np.array(self.blind_preprocessor.transform([4,8,4,0])).flatten()
                    }
            except:
                for trace in self.game_trace:
                    print(trace.decode("utf-8"))
                raise ValueError(
                    "Out of bounds",
                    i, self.action_masks[i], self.action_assignments, self.wrapped.obsDict()
                )
        return returned_dict
    def reset(self):
        orig_obs = self.wrapped.reset()
        self.update_avail_actions(orig_obs)
        return self.obsdict(orig_obs)
    def step(self, action_dict):
        printdict = {0: 'Adopt', 1: 'Override', 2: 'Wait', 3: 'Match', 4: 'Adopt Team', 5: 'Publish all'}
        currstr = ''.encode('utf-8')
        currstr += "Step: {0}\n".format(self.wrapped._accumulated_steps).encode('utf-8')
        for agent,action in action_dict.items():
            currstr += 'Agent {0} action {1}\n'.format(agent,printdict[action]).encode('utf-8')
            if self.action_masks[int(agent)][action] != 1:
                raise ValueError(
                    "Chosen action was not one of the unmasked ones",
                    agent,
                    action, self.action_masks[int(agent)],
                    self.action_assignments, self.wrapped.obsDict()
                )
        
        
        orig_obs, rew, done, info = self.wrapped.step(action_dict)
        currstr += "Rewards: {0}\n".format(self.wrapped._rewards).encode('utf-8')
        currstr += "Block receiver: {0}\n".format(self.wrapped._block_receiver).encode('utf-8')
        currstr += "Obs: {0}\n".format(orig_obs).encode('utf-8')
        tree = self.wrapped.convert_tree(self.wrapped._hidden_tree)
        currstr += "\n\n".encode('utf-8')
        currstr += drawTree2(False)(False)(tree).encode('utf-8')
        currstr += "\n".encode('utf-8')
        self.game_trace.appendleft(currstr)
        self.update_avail_actions(orig_obs)
        #print("Predicted total blocks", orig_obs["1"][2] + self.wrapped._accepted_blocks[1])
        #print("Actual total blocks", self.wrapped._total_blocks[1])
        #assert (orig_obs["1"][2] + self.wrapped._accepted_blocks[1] == self.wrapped._total_blocks[1])
        new_obs = self.obsdict(orig_obs)
        return new_obs, rew, done, info
        
def drawTree2(blnCompact):
    '''Monospaced UTF8 left-to-right text tree in a
       compact or expanded format, with any lines
       containing no nodes optionally pruned out.
    '''
    def go(blnPruned, tree):
        # measured :: a -> (Int, String)
        def measured(x):
            '''Value of a tree node
               tupled with string length.
            '''
            s = ' ' + str(x) + ' '
            return len(s), s
 
        # lmrFromStrings :: [String] -> ([String], String, [String])
        def lmrFromStrings(xs):
            '''Lefts, Mid, Rights.'''
            i = len(xs) // 2
            ls, rs = xs[0:i], xs[i:]
            return ls, rs[0], rs[1:]
 
        # stringsFromLMR :: ([String], String, [String]) -> [String]
        def stringsFromLMR(lmr):
            ls, m, rs = lmr
            return ls + [m] + rs
 
        # fghOverLMR
        # :: (String -> String)
        # -> (String -> String)
        # -> (String -> String)
        # -> ([String], String, [String])
        # -> ([String], String, [String])
        def fghOverLMR(f, g, h):
            def go(lmr):
                ls, m, rs = lmr
                return (
                    [f(x) for x in ls],
                    g(m),
                    [h(x) for x in rs]
                )
            return lambda lmr: go(lmr)
 
        # leftPad :: Int -> String -> String
        def leftPad(n):
            return lambda s: (' ' * n) + s
 
        # treeFix :: (Char, Char, Char) -> ([String], String, [String])
        #                               ->  [String]
        def treeFix(l, m, r):
            def cfix(x):
                return lambda xs: x + xs
            return compose(stringsFromLMR)(
                fghOverLMR(cfix(l), cfix(m), cfix(r))
            )
 
        def lmrBuild(w, f):
            def go(wsTree):
                nChars, x = wsTree['root']
                _x = ('─' * (w - nChars)) + x
                xs = wsTree['nest']
                lng = len(xs)
 
                # linked :: String -> String
                def linked(s):
                    c = s[0]
                    t = s[1:]
                    return _x + '┬' + t if '┌' == c else (
                        _x + '┤' + t if '│' == c else (
                            _x + '┼' + t if '├' == c else (
                                _x + '┴' + t
                            )
                        )
                    )
 
                # LEAF ------------------------------------
                if 0 == lng:
                    return ([], _x, [])
 
                # SINGLE CHILD ----------------------------
                elif 1 == lng:
                    def lineLinked(z):
                        return _x + '─' + z
                    rightAligned = leftPad(1 + w)
                    return fghOverLMR(
                        rightAligned,
                        lineLinked,
                        rightAligned
                    )(f(xs[0]))
 
                # CHILDREN --------------------------------
                else:
                    rightAligned = leftPad(w)
                    lmrs = [f(x) for x in xs]
                    return fghOverLMR(
                        rightAligned,
                        linked,
                        rightAligned
                    )(
                        lmrFromStrings(
                            intercalate([] if blnCompact else ['│'])(
                                [treeFix(' ', '┌', '│')(lmrs[0])] + [
                                    treeFix('│', '├', '│')(x) for x
                                    in lmrs[1:-1]
                                ] + [treeFix('│', '└', ' ')(lmrs[-1])]
                            )
                        )
                    )
            return lambda wsTree: go(wsTree)
 
        measuredTree = fmapTree(measured)(tree)
        levelWidths = reduce(
            lambda a, xs: a + [max(x[0] for x in xs)],
            levels(measuredTree),
            []
        )
        treeLines = stringsFromLMR(
            foldr(lmrBuild)(None)(levelWidths)(
                measuredTree
            )
        )
        return [
            s for s in treeLines
            if any(c not in '│ ' for c in s)
        ] if (not blnCompact and blnPruned) else treeLines
 
    return lambda blnPruned: (
        lambda tree: '\n'.join(go(blnPruned, tree))
    )
# Node :: a -> [Tree a] -> Tree a
def Node(v):
    '''Contructor for a Tree node which connects a
       value of some kind to a list of zero or
       more child trees.
    '''
    return lambda xs: {'type': 'Tree', 'root': v, 'nest': xs}
 
 
# compose (<<<) :: (b -> c) -> (a -> b) -> a -> c
def compose(g):
    '''Right to left function composition.'''
    return lambda f: lambda x: g(f(x))
 
 
# concatMap :: (a -> [b]) -> [a] -> [b]
def concatMap(f):
    '''A concatenated list over which a function has been mapped.
       The list monad can be derived by using a function f which
       wraps its output in a list,
       (using an empty list to represent computational failure).
    '''
    return lambda xs: list(
        chain.from_iterable(map(f, xs))
    )
 
 
# fmapTree :: (a -> b) -> Tree a -> Tree b
def fmapTree(f):
    '''A new tree holding the results of
       applying f to each root in
       the existing tree.
    '''
    def go(x):
        return Node(f(x['root']))(
            [go(v) for v in x['nest']]
        )
    return lambda tree: go(tree)
 
 
# foldr :: (a -> b -> b) -> b -> [a] -> b
def foldr(f):
    '''Right to left reduction of a list,
       using the binary operator f, and
       starting with an initial accumulator value.
    '''
    def g(x, a):
        return f(a, x)
    return lambda acc: lambda xs: reduce(
        g, xs[::-1], acc
    )
 
 
# intercalate :: [a] -> [[a]] -> [a]
# intercalate :: String -> [String] -> String
def intercalate(x):
    '''The concatenation of xs
       interspersed with copies of x.
    '''
    return lambda xs: x.join(xs) if isinstance(x, str) else list(
        chain.from_iterable(
            reduce(lambda a, v: a + [x, v], xs[1:], [xs[0]])
        )
    ) if xs else []
 
 
# iterate :: (a -> a) -> a -> Gen [a]
def iterate(f):
    '''An infinite list of repeated
       applications of f to x.
    '''
    def go(x):
        v = x
        while True:
            yield v
            v = f(v)
    return lambda x: go(x)
 
 
# levels :: Tree a -> [[a]]
def levels(tree):
    '''A list of the nodes at each level of the tree.'''
    return list(
        map_(map_(root))(
            takewhile(
                bool,
                iterate(concatMap(nest))(
                    [tree]
                )
            )
        )
    )
 
 
# map :: (a -> b) -> [a] -> [b]
def map_(f):
    '''The list obtained by applying f
       to each element of xs.
    '''
    return lambda xs: list(map(f, xs))
 
 
# nest :: Tree a -> [Tree a]
def nest(t):
    '''Accessor function for children of tree node.'''
    return t['nest'] if 'nest' in t else None
 
 
# root :: Tree a -> a
def root(t):
    '''Accessor function for data of tree node.'''
    return t['root'] if 'root' in t else None
            



        