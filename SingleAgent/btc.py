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
from environment import eth_env
from environment import SM_env_with_stale
from environment import random_normal_trunc
import mdptoolbox


class Qnetwork():
    def __init__(self, h_size, state_space_n, state_vector_n, action_space_n):

        self.state_space_n = state_space_n
        self.action_space_n = action_space_n
        self.state_vector_n = state_vector_n

        # The network recieves a state number from
        # It then resizes it and processes it through four convolutional layers.
        self.vectorIn = tf.placeholder(shape=[None, state_vector_n], dtype=tf.float32)
        #print(self.scalarInput)
        #self.vectorIn = tf.one_hot(self.scalarInput, state_space_n, dtype=tf.float32)
        #print(self.vectorIn)
        self.fc1 = tf.layers.dense(self.vectorIn, h_size, activation=tf.nn.relu)
        #print(self.fc1)
        self.fc2 = tf.layers.dense(self.fc1, h_size, activation=tf.nn.relu)
        #print(self.fc2)

        '''
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            biases_initializer=None)
        self.conv4 = slim.conv2d( \
            inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding='VALID',
            biases_initializer=None)
        '''

        # We take the output from the final layer and split it into separate advantage and value streams.
        #self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        #self.streamA = slim.flatten(self.streamAC)
        #self.streamV = slim.flatten(self.streamVC)

        #print(self.fc2)

        self.streamA, self.streamV = tf.split(self.fc2, 2, 1)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, action_space_n]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)
        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
        #print(self.Qout)
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, action_space_n, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

    def get_Q_table(self, sess, s):
        Q = sess.run(self.Qout, feed_dict={self.vectorIn:[s]})
        Q = np.reshape(Q, [-1])
        return Q

    def act_epsilon_greedy(self, sess, s, e = 0):

        #legal_move_list = env.legal_move_list(s)
        #legal_move_list = range(env._action_space_n)

        if np.random.rand(1) < e:
            a = np.random.choice(self.action_space_n)
        else:
            Q = self.get_Q_table(sess, s)
            a = np.argmax(Q)

            '''
            #print(Q)
            a = 0
            val = -100000
            for i in range(self.):
                if (Q[i] > val):
                    val = Q[i]
                    a = i
            '''

        return a

    def get_policy_table(self, sess, env):
        policy = np.zeros(self.state_space_n, dtype = np.int32)
        for i in range(0, self.state_space_n):
            ss = env._index_to_vector(i)
            policy[i] = self.act_epsilon_greedy(sess, ss, 0)
        return policy


class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        #print(experience)
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        #print(self.buffer)
        size = min(size, len(self.buffer))
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])
        #return random.sample(self.buffer, size)


#def processState(states):
#    return np.reshape(states, [21168])

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)


def LOAD_MODEL(path):
    input = open(path, "r")
    grid = int(input.readline())
    env = SM_env(max_hidden_block = 20, attacker_fraction = 0.4, follower_fraction = 0.5, dev = 0.0)
    policy = np.zeros((grid, env._state_space_n), dtype = np.int)
    for i in range(grid):
        #optimal_policy_all[i] = map(int(), input.readline()[0:-1].split(' '))
        policy[i] = list(map(int, input.readline().rstrip().split(' ')))
    input.close()
    return policy

# In most of the experiments, we only tune the following params :
num_episodes = 100 #How many episodes of game environment to train network with.
load_model = False #Whether to load a saved model.
load_best_model = False # load a saved best model
ALPHA = 0.4 # the hash power fraction of attacker
GAMMA = 0.5 # the follower's fraction
HIDDEN_BLOCK = 20 # maximum hidden block of attacker
rule = "longest" # "longest" -- bitcoin rule, "GHOST" -- GHOST rule
stale_rate = 0.0 # stale block rate, 0 means no stale block -- the classical selfish mining setting
DEV = 0.0 # the alpha's fluctuation rate, 0 means fixed alpha
know_alpha = True # if the agent knows the current alpha.
random_process = "iid" # or "brown" -- brownian process

#Other training params.
batch_size = 64 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 10000. #How many steps of training to reduce startE to endE.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 10000 #The max allowed length of our episode.
h_size = 100 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network
MDP_size_threshold = 10000 # limit the MDP size that the MDP solver can finished in reasonable time
interval = (0, 0.5) # the range of the alpha
rept = 100 # final test repetition time

path = "./btc_" + rule + str(HIDDEN_BLOCK) + "_" + str(h_size) #The path to save our model to.
if know_alpha == True: path += "know_alpha"
best_path = path + "/model_best.ckpt" # best model path
file = open("btc_" + rule + str(HIDDEN_BLOCK) + "_" + str(h_size) + ".txt", "a+")
file.write("\n\n\n\n A new test from now!")

env = SM_env_with_stale(max_hidden_block = HIDDEN_BLOCK, attacker_fraction = ALPHA, follower_fraction = GAMMA, rule = rule, stale_rate=stale_rate, dev = DEV, know_alpha = know_alpha, random_interval=interval, random_process = random_process, frequency=6)

if (env._state_space_n < MDP_size_threshold):
    env.MDP_matrix_init()
    P, R = env.get_MDP_matrix()
    solver = mdptoolbox.mdp.PolicyIteration(P, R, 0.99)
    solver.run()
    optimal_policy = solver.policy
    SM1_policy = np.zeros_like(optimal_policy)

tf.reset_default_graph()
mainQN = Qnetwork(h_size, env._state_space_n, env._state_vector_n, env._action_space_n)
targetQN = Qnetwork(h_size, env._state_space_n, env._state_vector_n, env._action_space_n)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)

myBuffer = experience_buffer()

#Set the rate of random action decrease.
e = startE
stepDrop = (startE - endE)/annealing_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
fList = []
total_steps = 0
history_best = 0

#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:

    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)

    if load_best_model == True:
        print('Loading Best Model..')
        saver.restore(sess, best_path)

    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        #Reset environment and get first new observation
        s = env.reset()

        d = False
        rAll = 0
        j = 0
        #The Q-Network
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1

            #Choose an action by greedily (with e chance of random action) from the Q-network
            #Force to choose from the legal move list

            a = 0
            if (total_steps < pre_train_steps):
                a = mainQN.act_epsilon_greedy(sess, s, 1) # random act
            else:
                a = mainQN.act_epsilon_greedy(sess, s, e) # epsilon-greedy


            s1,r,d,_ = env.step(s, a, move = True)
            #ss1 = env._index_to_vector(s1)

            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
            #episodeBuffer.add(ay([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.

            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop

                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                    #print(trainBatch)
                    # Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict, feed_dict={mainQN.vectorIn: np.reshape(np.vstack(trainBatch[:, 3]), [-1, env._state_vector_n])})
                    Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.vectorIn: np.reshape(np.vstack(trainBatch[:, 3]), [-1, env._state_vector_n])})
                    end_multiplier = -(trainBatch[:, 4] - 1)
                    doubleQ = Q2[range(batch_size), Q1]
                    targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                    # Update the network with our target values.
                    _ = sess.run(mainQN.updateModel, \
                                 feed_dict={mainQN.vectorIn: np.reshape(np.vstack(trainBatch[:, 0]), [-1, env._state_vector_n]), mainQN.targetQ: targetQ,
                                            mainQN.actions: trainBatch[:, 1]})

                    updateTarget(targetOps, sess)  # Update the target network toward the primary network.
            rAll += r
            s = s1

            if d == True:
                break

        myBuffer.add(episodeBuffer.buffer)
        print("round = ", i, "training steps = ", j, "reward = ", rAll, "frac = ", env.reward_fraction)
        print("round = ", i, "training steps = ", j, "reward = ", rAll, "frac = ", env.reward_fraction, file = file)
        #jList.append(j)
        rList.append(rAll)
        fList.append(env.reward_fraction)
        jList.append(j)

        #Periodically save the model.
        if i % 100 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.ckpt')
            print("Saved Model")

        if (env.reward_fraction > history_best):
            history_best = env.reward_fraction
            saver.save(sess, best_path)

        if len(rList) % 10 == 0:
            print(total_steps,np.mean(rList[-10:]), e)

        #saver.save(sess,path+'/model-'+str(i)+'.ckpt')

    if num_episodes > 0:
        print('history best = ', history_best)
        saver.restore(sess, best_path)

    ## map to legal action
    '''
    dqn_policy = mainQN.get_policy_table(sess, env)
    for i in range(dqn_policy.shape[0]):
        dqn_policy[i] = env.map_to_legal_action(i, dqn_policy[i])

    print("strategy")
    print("strategy", file = file)
    for i in range(env.observation_space_n):
        state = env._index_to_name(i)
        s = env._index_to_vector(i)

        Q = mainQN.get_Q_table(sess, s)
        a = dqn_policy[i]
        action = env.mapped_name_of_action(i, a)

        #if (abs(Q[a]) < 0.0001):
        #    action = "non-determined"

        #print(h1, " ", h2, " ", status)
        h1 = state[0]
        h2 = state[1]
        if (h1 < h2): a1 = 0
        elif (h1 == h2 and h1 == 1 and state[2] == "normal") : a1 = 0
        elif (h1 == h2 + 1 and h2 > 0): a1 = 1
        elif (h1 == HIDDEN_BLOCK + 1): a1 = 1
        else: a1 = 2

        SM1_policy[i] = a1

        print(state, "my=", action, "SM=", env.mapped_name_of_action(i, a1), "OSM=", env.mapped_name_of_action(i, optimal_policy[i]), Q[a], "Q-table:", Q)
        print(state, "my=", action, "SM=", env.mapped_name_of_action(i, a1), "OSM=", env.mapped_name_of_action(i, optimal_policy[i]), Q[a], "Q-table:", Q, file = file)
    '''

    #env = SM_env_with_stale(max_hidden_block = HIDDEN_BLOCK, attacker_fraction = ALPHA, follower_fraction = GAMMA, rule = rule, stale_rate=stale_rate, dev = DEV, know_alpha = know_alpha, random_interval=interval, rule = )
    env.seed(100)

    rept = 1
    avg = 0
    for t in range(rept):
        s = env.reset()
        for i in range(10000):
            #print(s)
            h1, h2, h3, status = s[0:4]
            if (h1 < h2): a = 0
            elif (h1 == h2 and h1 == 1 and status == 0) :
                #print("match")
                a = 0
            elif (h1 == h2 + 1 and h2 > 0): a = 1
            elif (h1 == HIDDEN_BLOCK + 1): a = 1
            else: a = 2
            s, r, d, _ = env.step(s, a, move = True)
        avg += env.reward_fraction / rept
    print("ALPHA = ", ALPHA, "GAMMA = ", GAMMA, "HIDDEN_BLOCK", HIDDEN_BLOCK)
    print("SM1 = ", avg)
    print("SM1 = ", avg, file = file)


    optimal_policy_all = LOAD_MODEL("optimal_policy.txt")
    aux_env = SM_env(max_hidden_block = 20, attacker_fraction = ALPHA, follower_fraction = GAMMA)

    env.seed(100)
    grid = 100
    avg = 0
    for i in range(rept):
        s = env.reset()
        for i in range(10000):
            #a = mainqn.act_epsilon_greedy(sess, s, 0)
            a = 0
            if (s[0] > 20 + 1): a = 1
            elif (s[1] > 20 + 1): a = 0
            else:
                ss = aux_env._vector_to_index((s[0], s[1], s[3]))
                a = optimal_policy_all[int(env._visible_alpha * grid), ss]
            #a = dqn_policy[s]
            s, r, d, _ = env.step(s, a, move = True)
        avg += env.reward_fraction / rept
    print("osm = ", avg)
    print("osm = ", avg, file = file)

    '''
    if (env._state_space_n < MDP_size_threshold):
        env.seed(100)
        avg = 0
        for i in range(rept):
            s = env.reset()
            for i in range(10000):
                a = optimal_policy[env._vector_to_index(s[0:4])]
                #a = dqn_policy[s]
                s, r, d, _ = env.step(s, a, move = True)
            avg += env.reward_fraction / rept
        print("MDP solver = ", avg)
        print("MDP solver = ", avg, file = file)

    # final simulation
    dqn_policy = np.zeros(env._state_space_n, dtype=np.int)

    for i in range(env._state_space_n):
        s = env._index_to_vector(i)
        a = mainQN.act_epsilon_greedy(sess, s, 0)
        dqn_policy[i] = env.map_to_legal_action(s, a)
    '''

    env.seed(100)
    avg = 0
    for i in range(rept):
        s = env.reset()
        for i in range(10000):
            #a = dqn_policy[env._vector_to_index(s)]
            a = mainQN.act_epsilon_greedy(sess, s, 0)
            #a = dqn_policy[s]
            s, r, d, _ = env.step(s, a, move = True)
        avg += env.reward_fraction / rept
    print("final simulation fraction = ", avg)
    print("final simulation fraction = ", avg, file = file)

    sess.close()


