'''
Credit to

https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0#.4gyadb8a4

This guy's tutorial is very helpful!


Handle the illegale move : map to a legal one
'''

from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
from environment import SM_env
from environment import random_normal_trunc
from environment import SM_env_with_stale
import mdptoolbox


def load_model(path):
    input = open(path, "r")
    grid = int(input.readline())
    env = SM_env(max_hidden_block = 20, attacker_fraction = 0.4, follower_fraction = 0.5, dev = 0.0)
    policy = np.zeros((grid, env._state_space_n), dtype = np.int)
    for i in range(grid):
        #optimal_policy_all[i] = map(int(), input.readline()[0:-1].split(' '))
        policy[i] = list(map(int, input.readline().rstrip().split(' ')))
    input.close()
    return policy

def SM1_policy():
    grid = 1
    cur_env = SM_env(max_hidden_block = HIDDEN_BLOCK, attacker_fraction = 0.4, follower_fraction = GAMMA, dev = 0)
    sm1_policy = np.zeros((grid, cur_env._state_space_n), dtype = np.int)
    for i in range(grid):
        for j in range(cur_env._state_space_n):
            h1, h2, status = cur_env._index_to_name(j)
            if (h1 < h2): a = 0
            elif (h1 == h2 and h1 == 1 and status == "normal") : a = 0
            elif (h1 == h2 + 1 and h2 > 0): a = 1
            elif (h1 == HIDDEN_BLOCK + 1): a = 1
            else: a = 2
            sm1_policy[i, j] = a
    return sm1_policy

def honest_policy():
    grid = 1
    cur_env = SM_env(max_hidden_block = HIDDEN_BLOCK, attacker_fraction = 0.4, follower_fraction = GAMMA, dev = 0)
    honest_policy = np.zeros((grid, cur_env._state_space_n), dtype = np.int)
    for i in range(grid):
        for j in range(cur_env._state_space_n):
            h1, h2, status = cur_env._index_to_name(j)
            if (h1 < h2): a = 0
            elif (h1 > h2): a = 1
            else: a = 2
            honest_policy[i, j] = a
    return honest_policy

def generate_optimal_model():
    env = SM_env(max_hidden_block = HIDDEN_BLOCK, attacker_fraction = ALPHA, follower_fraction = GAMMA)
    grid = 100
    optimal_policy_all = np.zeros((grid, env._state_space_n), dtype = np.int)
    for i in range(grid):
        cur_env = SM_env(max_hidden_block = HIDDEN_BLOCK, attacker_fraction = i * 1.0 / grid, follower_fraction = GAMMA, dev = 0)
        print(i)
        optimal_policy_all[i] = cur_env.optimal_mdp_solver()

    output = open("optimal_policy.txt", "w")
    print(grid, file = output)
    for i in range(grid):
        for j in range(optimal_policy_all.shape[1]):
            print(optimal_policy_all[i, j], end = " ", file = output)
        print(file = output)
    output.close()

    output = open("optimal_policy_visual.txt", "w")
    for i in range(optimal_policy_all.shape[1]):
        s = env._index_to_name(i)
        if (s[0] > 8 or s[1] > 8): continue
        print(s, end=" ", file = output)
        for j in range(grid):
            if (j == 0 or optimal_policy_all[j, i] != optimal_policy_all[j - 1, i]):
                print(j * (1.0 / grid), env.mapped_name_of_action(i, optimal_policy_all[j, i]), end=" ", file = output)
        print(file = output)
    output.close()
    return optimal_policy_all

def simulation(ALPHA, GAMMA, HIDDEN_BLOCK, DEV, policy, epoch, max_step, random_interval = (0, 1), random_process = "iid", array = []):
    #print(len(array))
    env = SM_env(max_hidden_block = HIDDEN_BLOCK, attacker_fraction = ALPHA, follower_fraction = GAMMA, dev = DEV, random_interval = random_interval, frequency = 6, random_process = random_process, array = array)
    #print(random_process)

    env.seed(1)
    grid = policy.shape[0]
    avg = 0
    for t in range(epoch):
        s = env.reset()
        for i in range(max_step):
            cur_alpha = env._current_alpha
            a = policy[min(int(cur_alpha * grid), grid - 1), s]
            s, r, d, _ = env.step(s, a, move = True)
        avg += env.reward_fraction / epoch
    return avg, env._expected_alpha

'''
def simulation(ALPHA, GAMMA, HIDDEN_BLOCK, DEV, policy, epoch, max_step, random_interval = (0, 1)):
    env = SM_env(max_hidden_block = HIDDEN_BLOCK, attacker_fraction = ALPHA, follower_fraction = GAMMA, dev = DEV, random_interval = random_interval, frequency = 12)

    grid = policy.shape[0]
    avg = 0
    for t in range(epoch):
        s = env.reset()
        for i in range(max_step):
            cur_alpha = env._current_alpha
            a = policy[min(int(cur_alpha * grid), grid - 1), s]
            s, r, d, _ = env.step(s, a, move = True)
        avg += env.reward_fraction / epoch
    return avg, env._expected_alpha
'''

ALPHA = 0.4 # the hash power fraction of attacker
GAMMA = 0.5 # the follower's fraction
HIDDEN_BLOCK = 20 # maximum hidden block of attacker
DEV = 0.1
epoch = 100
max_step = 10000
LOAD_OPTIMAL = False

if (LOAD_OPTIMAL == True):
    optimal_policy_all = load_model("optimal_policy.txt")
else:
    optimal_policy_all = generate_optimal_model()
exit()

dqn_policy_all = load_model("dqn_policy.txt")

sm1_policy = SM1_policy()
honest_policy = honest_policy()

'''
avg = simulation(ALPHA, GAMMA, HIDDEN_BLOCK, DEV, optimal_policy_all, epoch, max_step)
file = open("optimal_result.txt", "a+")
print("OSM know alpha fraction = ", avg)
print("OSM know alpha fraction = ", avg, file = file)

avg = simulation(ALPHA, GAMMA, HIDDEN_BLOCK, DEV, dqn_policy_all, epoch, max_step)
print("RL know alpha fraction = ", avg)
print("RL know alpha fraction = ", avg, file = file)
file.close()
'''
interval = (0, 0.5)

'''
r1, a1 = simulation(ALPHA, GAMMA, HIDDEN_BLOCK, 0.1, optimal_policy_all, epoch, max_step, interval)
r2, a2 = simulation(ALPHA, GAMMA, HIDDEN_BLOCK, 0.1, dqn_policy_all, epoch, max_step, interval)
print(ALPHA, interval, "mean = ", a1)
print("OSM = ", r1, "RL = ", r2)
exit()
'''

dev_array = np.array([0, 0.1, 0.2])
'''
for i in range(dev_array.shape[0]):
    file = open("./Result/up" + str(interval[1]) + "dev" + str(dev_array[i]) +".csv", "w")
    print("alpha, gamma, dev, E[alpha], OSM, RL, SM1", file = file)
    print("alpha, gamma, dev, E[alpha], OSM, RL, SM1")
    alpha_array = np.linspace(0.1, 0.5, 20)
    for j in range(alpha_array.shape[0]):
        r1, a1 = simulation(alpha_array[j], GAMMA, HIDDEN_BLOCK, dev_array[i], optimal_policy_all, epoch, max_step, interval)
        r2, a2 = simulation(alpha_array[j], GAMMA, HIDDEN_BLOCK, dev_array[i], dqn_policy_all, epoch, max_step, interval)
        r3, a3 = simulation(alpha_array[j], GAMMA, HIDDEN_BLOCK, dev_array[i], sm1_policy, epoch, max_step, interval)
        print("%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f" % (alpha_array[j], GAMMA, dev_array[i], a1, r1, r2, r3), file = file)
        print("%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f" % (alpha_array[j], GAMMA, dev_array[i], a1, r1, r2, r3))
    file.close()
'''
#alpha_array = np.array([0.25, 0.3, 0.35, 0.4])
alpha_array = np.array([0.4])
random_process = "iid"
for i in range(alpha_array.shape[0]):
    file = open("./Result/up" + str(interval[1]) + "dev" + str(dev_array[i]) +".csv", "w")
    print("alpha, gamma, dev, E[alpha], OSM, RL, SM1, honest", file = file)
    print("alpha, gamma, dev, E[alpha], OSM, RL, SM1, honest")
    dev_array = np.linspace(0, 0.2, 10)
    for j in range(dev_array.shape[0]):
        r1, a1 = simulation(alpha_array[i], GAMMA, HIDDEN_BLOCK, dev_array[j], optimal_policy_all, epoch, max_step, interval, random_process)
        #r2, a2 = simulation(alpha_array[i], GAMMA, HIDDEN_BLOCK, dev_array[j], dqn_policy_all, epoch, max_step, interval, random_process)
        r2, a2 = 0, 0
        r3, a3 = simulation(alpha_array[i], GAMMA, HIDDEN_BLOCK, dev_array[j], sm1_policy, epoch, max_step, interval, random_process)
        r4, a4 = simulation(alpha_array[i], GAMMA, HIDDEN_BLOCK, dev_array[j], honest_policy, epoch, max_step, interval, random_process)
        print("%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f" % (alpha_array[i], GAMMA, dev_array[j], a1, r1, r2, r3, r4), file = file)
        print("%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f" % (alpha_array[i], GAMMA, dev_array[j], a1, r1, r2, r3, r4))
    file.close()

#print("OSM fraction = ", env.theoretical_attacker_fraction(optimal_policy))
#print("OSM fraction = ", env.theoretical_attacker_fraction(optimal_policy), file = file)


