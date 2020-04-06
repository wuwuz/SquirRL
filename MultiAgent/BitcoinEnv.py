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
import tree_draw_fcns

class BlockNode:
    def __init__(self, N):
        # who owns this block
        self.owner = None
        # what the block is referencing
        self.parent = None
        # who has adopted the block
        self.adopted = set()
        # the length from the root. we update this every time we get full adoption
        self.length = 0
        # the children
        self.children = []
        # number of blocks that each attacker owns up to this point, plus the honest nodes
        self.owns = [0 for i in range(N + 1)]

class BitcoinEnv(MultiAgentEnv):
    """Two-player environment for rock paper scissors.
    The observation is simply the last opponent action."""

    def __init__(self, env_config):
        
        self.max_states = env_config['max_hidden_block'] + 1

        
        self._alphas = env_config['alphas']
        self._gammas = env_config['gammas']
        self._print_trees = env_config['print']
        self._spy = env_config['spy']
        self._team_spirit = env_config['team_spirit']
        self.action_space = spaces.Discrete(6)
        self.observation_space = constants.make_spy_space(len(self._alphas), self.max_states - 1)
        self._N = len(self._alphas)
        self._honest_power = 1 - sum(self._alphas)
        self._attacker_lengths = [0 for i in range(self._N)]
        # initialize to a dummy node, makes things easier
        self._hidden_tree = BlockNode(self._N) 
        self._episode_length = env_config['ep_length']
        # everyone adopts the root
        self._hidden_tree.adopted.update(range(self._N + 1))
        # the last place each attacker last adopted
        self._fork_locations = [self._hidden_tree for i in range(self._N)]

        # the place(s) (plural to account for forking, this can be as large as N) where the honest nodes will mine from
        self._honest_locations = [self._hidden_tree]
        self._accumulated_steps = 0
        self._accepted_blocks = [0 for i in range(self._N + 1)]
        self.current_visible_state = [[0,0,0,"normal"] for i in range(self._N)]
        self._total_blocks = [0 for i in range(self._N + 1)]
        self._actions_taken = [[0,0,0,0,0] for i in range(self._N)]
    def reset(self):
        self._current_alpha = self._alphas
        self._accumulated_steps = 0
        self._actions_taken = [[0,0,0,0,0,0] for i in range(self._N)]
        self._hidden_tree = BlockNode(self._N) # initialize to a dummy node, makes things easier
        # everyone adopts the root
        self._hidden_tree.adopted.update(range(self._N + 1))

        # the last place each attacker adopted
        self._fork_locations = [self._hidden_tree for i in range(self._N)]

        # the place(s) (plural to account for forking, this can be as large as N) where the honest nodes will mine from
        self._honest_locations = [self._hidden_tree]
        # 0th coordinate: length of current branch
        # 1st coordinate: length of longest branch (e.g. length of the "honest" chain)
        # 2nd coordinate: length of the current branch
        # 3rd coordinate: forking behavior wrt the "main chain"
        self._current_visible_state = [[0,0,0, "normal"] for i in range(self._N)]
        self._accepted_blocks = [0 for i in range(self._N + 1)]
        self._total_blocks = [0 for i in range(self._N + 1)]

        obs = self.obsDict()
        returnedDict = obs

        return returnedDict
    def convert_tree(self, root):
        children = []
        for child in root.children:
            children.append(self.convert_tree(child))
        return {'type': 'Tree', 'root': (root.owner, list(root.adopted).copy(), root.length, root.owns.copy()), 'nest':children}

    def map_action_fix(self, a, b, status, action, i):
        if a >= self.max_states or b >= self.max_states:
            if a > b:
                return 1
            else:
                return 0
        elif self._current_visible_state[i][0] > self._fork_locations[i].length:
            return 5
        else:
            return 4
        return action
    
    def map_action(self, a, b, status, action, i):
        if a >= self.max_states or b >= self.max_states:
            if a > b:
                return 5
            else:
                return 0
        elif self._fork_locations[i].length >= self._current_visible_state[i][0]:
            if action == 1 or action == 3 or action == 5:
                return 4
        elif a < b:
            if action == 1 or action == 3:
                return 4
        elif a == b and a == 0:
            if action == 1 or action == 3:
                return 4
        elif a == b and status == 'normal':
            if action == 1:
                return 4
        elif a == b and (status == 'catch up' or status == 'forking'):
            if action == 1 or action == 3:
                return 4
        elif a > b and b == 0:
            if action == 0 or action == 3:
                return 5
        elif a > b and b > 0 and status == 'normal':
            if action == 0:
                return 5
        elif a > b and b > 0 and (status == 'forking' or status == 'catch_up'):
            if action == 0 or action == 3:
                return 5
        else:
            '''
            if action == 1:
                return 5
            elif action == 0:
                return 4
            '''
            return action
        '''
        if action == 1:
            return 5
        elif action == 0:
            return 4
        '''
        return action
    
    def step(self, action_dict):
        #print("step", self._accumulated_steps)
        event = np.random.choice(len(self._alphas) + 1, p = self._alphas + [self._honest_power])
        self._total_blocks[event] += 1
        overrideFlag = False
        adoptFlags = []
        adoptTeam = []
        next_honest_locations = set(self._honest_locations)
        if self._print_trees:
            print('Step', self._accumulated_steps)
        printdict = {0: 'Adopt', 1: 'Override', 2: 'Wait', 3: 'Match', 4: 'Adopt Team', 5: 'Publish all'}
        for agent in action_dict.keys():
            i = int(agent)
            fork_location = self._fork_locations[i]
            a = self._current_visible_state[i][0]
            b = self._current_visible_state[i][1]
            status = self._current_visible_state[i][3]
            action = self.map_action(a, b, status, action_dict[agent], i)
            #print(str(i), action)
            self._actions_taken[i][action] += 1
            if self._print_trees:
                print('Agent ' + agent + ' action', printdict[action])
            if action == 0:
                adoptFlags.append(agent)
            elif action == 1:
                honest_location = self._honest_locations[np.random.randint(len(self._honest_locations), size = 1)[0]]
                self._fork_locations[i] = self.block_update(honest_location, fork_location, i, 1)
                if not overrideFlag:
                    next_honest_locations = set()
                    next_honest_locations.add(self._fork_locations[i])
                    overrideFlag = True
                else:
                    next_honest_locations.add(self._fork_locations[i])
            elif action == 2:
                continue
            elif action == 3:
                honest_location = self._honest_locations[np.random.randint(len(self._honest_locations), size = 1)[0]]
                self._fork_locations[i] = self.block_update(honest_location, fork_location, i, 0)
                if not overrideFlag:
                    next_honest_locations.add(self._fork_locations[i])
            elif action == 4:
                adoptTeam.append(agent)
            elif action == 5:
                leng = self._current_visible_state[i][0] - self._fork_locations[i].length
                self._fork_locations[i] = self.block_update(fork_location, fork_location, i, leng)
                next_honest_locations.add(self._fork_locations[i])
        
        max_length = 0
        for loc in next_honest_locations:
            if loc.length > max_length:
                max_length = loc.length
        self._honest_locations = []
        for loc in next_honest_locations:
            if loc.length == max_length:
                self._honest_locations.append(loc)
        

        for agent in adoptFlags:
            i = int(agent)
            chosen_random = np.random.randint(len(self._honest_locations), size = 1)[0]
            self.abandon(self._fork_locations[i], i)
            self._fork_locations[i] = self._honest_locations[chosen_random]
            self.adopt(self._fork_locations[i], i)
            self._current_visible_state[i][0] = self._honest_locations[0].length
            self._current_visible_state[i][3] = "catch up"
        
        fork_locations_lengths = [self._fork_locations[i].length for i in range(self._N)]
        forkmax = np.argmax(fork_locations_lengths)
        for agent in adoptTeam:
            i = int(agent)
            self.abandon(self._fork_locations[i], i)
            self._fork_locations[i] = self._fork_locations[forkmax]
            self.adopt(self._fork_locations[i], i)
            self._current_visible_state[i][0] = self._fork_locations[i].length
            self._current_visible_state[i][3] = "catch up"
        # prune the tree for rewards
        rewards = [0 for i in range(self._N + 1)]
        if len(self._honest_locations) == 1:
            currnode = self._honest_locations[0]
            root_candidates = set()
            while currnode != None:                        
                currnode.adopted.add(self._N)
                if len(currnode.adopted) == self._N + 1 and currnode != self._hidden_tree:
                    root_candidates.add(currnode)
                    break
                currnode = currnode.parent
            root_candidates = list(root_candidates)
            if len(root_candidates) > 0:
                # prune the tree, give out rewards
                root_lengths = np.asarray([root.length for root in root_candidates])
                root_idx = np.argmax(root_lengths)
                rewards = root_candidates[root_idx].owns.copy()
                root_candidates[root_idx].parent = None
                root_length = root_candidates[root_idx].length
                for i in range(len(self._attacker_lengths)):
                    self._current_visible_state[i][0] -= root_length
                new_owns = [0 for i in range(self._N + 1)]
                self.prune_update(root_candidates[root_idx], root_length, new_owns)
                self._hidden_tree = root_candidates[root_idx]
        honest_set = set(self._honest_locations)
        # handle the block distribution here as well as fork status changes
        # fork status is permanent until game is or the player adopts
        if len(honest_set) > 1: 
            for i in range(self._N):
                if self._fork_locations[i] in honest_set and self._fork_locations[i].owner == i:
                    self._current_visible_state[i][3] = "forking"
        else:
            for i in range(self._N):
                self._current_visible_state[i][3] = "normal"
        # if one of the honest nodes get it
        if event == self._N:
            leaders = []
            follower_fractions = []
            honestflag = False
            id_to_fork_reference = dict()
            for node in self._honest_locations:
                if node.owner == self._N:
                    honestflag = True
                    id_to_fork_reference[self._N] = node
                elif node.owner != None:
                    id_to_fork_reference[node.owner] = node  
                    follower_fractions.append(self._gammas[node.owner])
                    leaders.append(node.owner)
                else:
                    id_to_fork_reference[-1] = self._honest_locations[0]
            
            
            # if one of the forks has honest party at the end, then its follower fraction is
            # everything other than the follower fractions of the attackers
            if honestflag:
                honest_fraction = 1 - sum(follower_fractions)
                leaders.append(self._N)
                follower_fractions.append(honest_fraction)
                
            # the case when we're at the very beginning, no owners
            elif len(leaders) == 0:
                leaders = [-1]
                follower_fractions = [1]
            # if not, we have to divide the non-followers evenly accross the forks
            else:
                honest_fraction = 1 - sum(follower_fractions)
                marginal_increase = honest_fraction/len(leaders)
                follower_fractions = [x + marginal_increase for x in follower_fractions]
            chosen_fork = np.random.choice(leaders, p = follower_fractions)
            curr_mine_location = id_to_fork_reference[chosen_fork]

            # build on this location
            newNode = BlockNode(self._N)
            newNode.owner = self._N
            newNode.length = curr_mine_location.length + 1
            newNode.parent = curr_mine_location

            self.adopt(newNode, self._N)

            newNode.owns = curr_mine_location.owns.copy()
            newNode.owns[newNode.owner] += 1

            curr_mine_location.children.append(newNode)
            # this is now the new longest chain
            self._honest_locations = [newNode]


            for i in range(self._N):
                if self._current_visible_state[i][3] != "forking":
                    self._current_visible_state[i][3] = "normal"
        # if someone else mines the block, then it just goes on their private fork
        else:
            self._current_visible_state[event][0] += 1
            # no match allowed if someone else mined the last block
        # update the visible states
        for i in range(self._N):
            other_owns = sum(self._fork_locations[i].owns)
            other_owns = other_owns - self._fork_locations[i].owns[i]
            self._current_visible_state[i][1] = self._honest_locations[0].length 
            self._current_visible_state[i][2] = self._current_visible_state[i][0] - other_owns
        if event != self._N:
            for i in range(self._N):
                if i == event:
                    if self._current_visible_state[event][0] == self._honest_locations[0].length:
                        self._current_visible_state[event][3] = "catch up"
                    elif self._current_visible_state[event][3] != "forking":
                        self._current_visible_state[event][3] = "normal"
                    # if it is not a fork, it leads into 'normal'
                else:
                    # normal leads to normal, fork leads to fork, catch up is the one we need logic for
                    if self._current_visible_state[event][3] == "catch up":
                        if self._current_visible_state[event][0] != self._current_visible_state[event][1]:
                            self._current_visible_state[event][3] = "normal"
        transformed_rewards = [0. for i in range(self._N)]
        self._accepted_blocks = np.asarray(self._accepted_blocks) + np.asarray(rewards)
        self._accumulated_steps += 1
        
        total_selfish_alpha = np.sum(self._alphas)
        total_selfish_transf_rew = (1 - total_selfish_alpha)*np.sum(rewards[:-1]) - total_selfish_alpha*rewards[-1]

        for i in range(self._N):
            positive_reward = (1 - self._alphas[i])*rewards[i]
            negative_reward = -self._alphas[i]*(np.sum(rewards) - rewards[i])
            final_reward = positive_reward + negative_reward
            transformed_rewards[i] = final_reward
        done = {
            "__all__": self._accumulated_steps >= self._episode_length
        }
        rew = dict()
        for i in range(len(transformed_rewards)):
            rew[str(i)] = transformed_rewards[i]*(1 - self._team_spirit) + self._team_spirit*total_selfish_transf_rew
        info = dict()
        for i in range(len(transformed_rewards)):
            curr_dict = dict()
            won_blocks = self._accepted_blocks[i]
            total_blocks = np.sum(self._accepted_blocks)
            curr_dict['Won blocks'] = won_blocks
            curr_dict['Total blocks'] = total_blocks
            for k in range(6):
                curr_dict[str(k)] = self._actions_taken[i][k]/self._accumulated_steps
            info[str(i)] = curr_dict 
        obs = self.obsDict()
        if self._print_trees:
            print('Rewards', rewards)
            print('Block receiver', event)
            print("Obs", obs)
            tree = self.convert_tree(self._hidden_tree)
            print('\n')
            print(drawTree2(False)(False)(tree))
            print('\n')
        return obs, rew, done, info
    # update the publicly available tree when overriding
    # honest node is the current longest node
    # attacker_node is the place that the attacker is forking from
    def block_update(self, honest_node, attacker_node, attacker_id, extra_block = 0):
        honest_length = honest_node.length
        attacker_length = attacker_node.length
        nodes_to_make = honest_length - attacker_length + extra_block
        
        currnode = attacker_node
        for i in range(nodes_to_make):
            newNode = BlockNode(self._N)
            currnode.children.append(newNode)
            newNode.owner = attacker_id
            newNode.parent = currnode
            newNode.adopted.add(attacker_id)
            newNode.length = currnode.length + 1
            newNode.owns = currnode.owns.copy()
            newNode.owns[attacker_id] += 1

            currnode = newNode
        
        return currnode

    def obsDict(self):
        returnedDict = dict()
        private_chains = [self._fork_locations[i].length for i in range(len(self._alphas))]
        for i in range(len(self._current_visible_state)):
            if self._spy[i] == 1:
                additional_info = private_chains
                if self._current_visible_state[i][3] == 'normal':
                    returnedDict[str(i)] = tuple(self._current_visible_state[i][:3] + [constants.NORMAL] + additional_info + self._fork_locations[i].owns.copy())
                elif self._current_visible_state[i][3] == 'forking':
                    returnedDict[str(i)] = tuple(self._current_visible_state[i][:3] + [constants.FORKING] + additional_info + self._fork_locations[i].owns.copy())
                else:
                    returnedDict[str(i)] = tuple(self._current_visible_state[i][:3] + [constants.CATCH_UP] + additional_info + self._fork_locations[i].owns.copy())
            else:
                additional_info = []
                if self._current_visible_state[i][3] == 'normal':
                    returnedDict[str(i)] = tuple(self._current_visible_state[i][:3] + [constants.NORMAL] + additional_info)
                elif self._current_visible_state[i][3] == 'forking':
                    returnedDict[str(i)] = tuple(self._current_visible_state[i][:3] + [constants.FORKING] + additional_info)
                else:
                    returnedDict[str(i)] = tuple(self._current_visible_state[i][:3] + [constants.CATCH_UP] + additional_info)
            
        return returnedDict
    # update the lengths post-pruning
    # update the nodes that are owned up to this point
    def prune_update(self, root, length, seen):
        newseen = seen.copy()
        if root.parent != None:
            newseen[root.owner] += 1
        root.owns = newseen
        root.length -= length
        for node in root.children:
            self.prune_update(node, length, newseen)
    def adopt(self, root, id):
        if root == None:
            return
        root.adopted.add(id)
        self.adopt(root.parent, id)
    def abandon(self, root, id):
        if root == None:
            return
        root.adopted.remove(id)
        self.abandon(root.parent, id)
# visualization functions
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