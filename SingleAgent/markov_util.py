import numpy as np
import mdptoolbox

def null(A, eps=1e-8):
    u, s, vh = np.linalg.svd(A)
    null_space = np.compress(s <= eps, vh, axis=0)
    return null_space.T

## Markov Process

# input : transition matrix A(n x n)
# return : stationary distribution p(1xn)
# pA = p
# p(A-I) = 0
# (A-I)^T p^T = 0

def MP_stationary_distribution(A):
    #A = np.array(A, dtype= np.double)
    n = A.shape[0]
    I = np.eye(n, dtype = np.float32)
    #p = np.linalg.solve((A - I).T, np.zeros(n))
    p = null((A - I).T).T
    p = p.reshape(n)
    p = p / np.sum(p)
    #print(p.shape)
    #print(p)
    return p

## Markov Reward Process
# input : transition matrix A(n x n), reward matrix R(n x n)
# output : expected reward

def MRP_expected_reward(A, R):
    A = np.array(A, dtype=np.float32)
    R = np.array(R, dtype=np.float32)

    #mdptoolbox.util.checkSquareStochastic(A)
    #print(A)
    #print(R)
    n = A.shape[0]

    eps = 1e-6
    for i in range(n):
        cnt = 0
        for j in range(n):
            if (A[i,j] < 0):
                print("false", i, j, A[i, j])
            cnt += A[i, j]
        if (abs(cnt - 1) > eps):
            print(">1", i)
    print("pass")

    p = MP_stationary_distribution(A)
    expected_reward = 0.0
    for i in range(n):
        for j in range(n):
            expected_reward += p[i] * 1.0 * A[i, j] * R[i, j]
    #print(expected_reward)
    return expected_reward
