import os
import numpy as np 
import math
'''
root = '/afs/ece.cmu.edu/usr/charlieh/eval_results'
setting = "RL2"
team_spirit = 0
players = 2
spy = 0
alpha = .45
pairs = []
for iteration in range(1, 205):
    picklename = "{setting}_{spirit}_{blocks}_{alpha:04d}_{spy}_{iteration}.npy".format(spirit = int(team_spirit*100), 
            blocks = 5, 
            alpha = int(alpha*10000), 
            spy = spy,
            setting = setting,
            iteration = iteration)
    if os.path.isfile(os.path.join(root, picklename)):
        res = np.load(os.path.join(root, picklename), allow_pickle=True)
        pairs.append((iteration, res))
pairs.sort(key=lambda x: x[0])
print(pairs)
np.save(os.path.join('/afs/ece.cmu.edu/usr/charlieh/compiled_results',setting), pairs, allow_pickle=True)
'''
'''
root = '/afs/ece.cmu.edu/usr/charlieh/eval_results'
setting = "OSMvsOSM"
team_spirit = 0
players = 2
spy = 0
#iteration = 204
pairs = []
for alpha in np.linspace(.05, .45, 9):
    picklename = "{setting}_{spirit}_{blocks}_{alpha:04d}_{spy}_{iteration}.npy".format(spirit = int(team_spirit*100), 
            blocks = 5, 
            alpha = int(alpha*10000), 
            spy = spy,
            setting = setting,
            iteration = 'osmvsosm')
    if os.path.isfile(os.path.join(root, picklename)):
        res = np.load(os.path.join(root, picklename), allow_pickle=True)
        pairs.append((alpha, res))
pairs.sort(key=lambda x: x[0])
print(pairs)
np.save(os.path.join('/afs/ece.cmu.edu/usr/charlieh/compiled_results',setting), pairs, allow_pickle=True)
'''
spy = 0
team_spirit = 0
root = '/afs/ece.cmu.edu/usr/charlieh/eval_results'
for players in range(1,2):
    setting = "RL{0}".format(players)
    pairs = []
    for i in np.linspace(.05, .45, 9):        
        iteration = 0
        for item2 in os.listdir(root):
            item2split = item2.split('_')
            if item2split[0] == "RL{0}".format(players + 1) and int(i*10000) == int(item2split[3]):
                iteration = max(iteration, int(item2split[-1].split('.')[0]))
        picklename = "{setting}_{spirit}_{blocks}_{alpha:04d}_{spy}_{iteration}.npy".format(spirit = int(team_spirit*100), 
        blocks = 5, 
        alpha = int(i*10000), 
        spy = spy,
        setting = "RL{0}".format(players + 1),
        iteration = iteration)
        res = np.load(os.path.join(root, picklename), allow_pickle=True)
        pairs.append((i/players, res))
    pairs.sort(key=lambda x: x[0])
    print(pairs)
    np.save(os.path.join('/afs/ece.cmu.edu/usr/charlieh/compiled_results',setting), pairs, allow_pickle=True)
for players in range(2,4):
    setting = "RL{0}".format(players)
    pairs = []
    for i in np.linspace(.05, .99, 9):        
        iteration = 0
        for item2 in os.listdir(root):
            item2split = item2.split('_')
            if item2split[0] == "RL{0}".format(players + 1) and int(i*10000/players) == int(item2split[3]):
                iteration = max(iteration, int(item2split[-1].split('.')[0]))
        picklename = "{setting}_{spirit}_{blocks}_{alpha:04d}_{spy}_{iteration}.npy".format(spirit = int(team_spirit*100), 
        blocks = 5, 
        alpha = int(i*10000/(players)), 
        spy = spy,
        setting = "RL{0}".format(players + 1),
        iteration = iteration)
        res = np.load(os.path.join(root, picklename), allow_pickle=True)
        pairs.append((i/players, res))
    pairs.sort(key=lambda x: x[0])
    print(pairs)
    np.save(os.path.join('/afs/ece.cmu.edu/usr/charlieh/compiled_results',setting), pairs, allow_pickle=True)    
