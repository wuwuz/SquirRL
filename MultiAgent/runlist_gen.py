import os
import numpy as np
'''
root = '/afs/ece.cmu.edu/usr/charlieh/ray_results/PPO/'
with open('block_eval_args.txt', 'w+') as f:
    for item in os.listdir(root):
        currpath = os.path.join(root, item)
        if os.path.isdir(currpath):
            dirsplit = currpath.split('_')
            block_num = int(dirsplit[4])
            for item2 in os.listdir(currpath):
                currpath2 = os.path.join(currpath, item2)
                if os.path.isdir(currpath2):
                    item2split = item2.split('_')
                    f.write("{0} {1}".format(block_num, os.path.join(currpath2, 
                        item2split[0] + '-' + item2split[1])) + '\n')
'''
'''
with open('OSM_args.txt', 'w+') as f:
    for i in np.linspace(.05, .5, 9):
        f.write("{blocks} {alphas:.04f} {spy} {players} {osm} {savepath} {extended} {teamspirit}\n".format(
                blocks = 5,
                alphas = round(i, 4),
                spy = 0,
                players = 2,
                osm = 1,
                savepath = "none",
                extended = 0,
                teamspirit = 0
            ))
'''
'''
with open('OSMvsOSM_args.txt', 'w+') as f:
    for i in np.linspace(.05, .45, 9):
        f.write("{blocks} {alphas:.04f} {spy} {players} {osm} {savepath} {extended} {teamspirit}\n".format(
                blocks = 5,
                alphas = round(i, 4),
                spy = 0,
                players = 2,
                osm = 2,
                savepath = "osmvsosm",
                extended = 0,
                teamspirit = 0
            ))
'''
'''
with open('RL2_args.txt', 'w+') as f:
    for i in np.linspace(.05, .45, 9):
        f.write("{blocks} {alphas:.04f} {spy} {players} {osm} {savepath} {extended} {teamspirit}\n".format(
                blocks = 5,
                alphas = round(i, 4),
                spy = 0,
                players = 2,
                osm = 0,
                savepath = "none",
                extended = 0,
                teamspirit = 0
            ))
'''
'''
with open('RL3_args.txt', 'w+') as f:
    for i in np.linspace(.30, .33, 9):
        f.write("{0} {1} {2} {3} {4} {5}\n".format(
            5,
            round(i, 4),
            0,
            3,
            0,
            "none"
        ))
'''
'''
with open('RL5.txt', 'w+') as f:
    spy = 0
    extended = 0
    
    for players in range(1,2):
        for i in np.linspace(.05, .5, 9):
            f.write("{blocks} {alphas:.04f} {spy} {players} {osm} {savepath} {extended} {teamspirit}\n".format(
                blocks = 5,
                alphas = round(i/players, 4),
                spy = spy,
                players = players,
                osm = 0,
                savepath = "none",
                extended = extended,
                teamspirit = 0
            ))
    
    for players in range(2,6):
        for i in np.linspace(.05, .99, 9):
            f.write("{blocks} {alphas:.04f} {spy} {players} {osm} {savepath} {extended} {teamspirit}\n".format(
                blocks = 5,
                alphas = round(i/players, 4),
                spy = spy,
                players = players,
                osm = 0,
                savepath = "none",
                extended = extended,
                teamspirit = 0
            ))
'''
with open('fedgan_input.txt', 'w+') as f:
    for i in range(21):
        f.write("{het} {het} {eta} {eta} {tau}\n".format(het = i, eta = 0.1, tau = 1))
    for i in range(21):
        f.write("{het} {het} {eta} {eta} {tau}\n".format(het = i, eta = 0.01, tau = 1))
    for i in range(21):
        f.write("{het} {het} {eta} {eta} {tau}\n".format(het = i, eta = 0.05, tau = 1))
'''
with open('train_spirit.txt', 'w+') as f:
    for i in np.linspace(0, 1, 11):
        f.write("{blocks} {alphas:.04f} {spy} {players} {osm} {savepath} {extended} {teamspirit}\n".format(
                blocks = 5,
                alphas = 0.3,
                spy = 1,
                players = 3,
                osm = 0,
                savepath = "none",
                extended = 1,
                teamspirit = i
            ))
'''

'''
setting_sets = ["RLvsOSM", "RL2", "RL3", "RL4"]
setting_sets = ["RLvsOSM"]
setting_sets = ["RL3"]
runfile = 'RL3_cont.txt'  
root = '/afs/ece.cmu.edu/usr/charlieh/ray_results/RLX_v2'
players = {"RL1":1,"RLvsOSM": 2, "RL2":2, "RL3":3, "RL4":4}
osm = {"RLvsOSM": 1, "RL2":0, "RL3":0, "RL4":0, "RL1": 0}
with open(runfile, 'w+') as f:
    for setting_set in setting_sets:
        for item in os.listdir(root):
            currpath = os.path.join(root, item)
            if os.path.isdir(currpath):
                dirsplit = item.split('_')
                alpha = int(dirsplit[4])
                setting = dirsplit[1]
                if setting != setting_set:
                    continue
                
                iteration = 0
                for item2 in os.listdir(currpath):
                    currpath2 = os.path.join(currpath, item2)
                    if os.path.isdir(currpath2):
                        item2split = item2.split('_')
                        iteration = max(iteration, int(item2split[1]))
                f.write("{blocks} {alphas:.04f} {spy} {players} {osm} {savepath} {extended} {teamspirit}\n".format(
                        blocks = 5,
                        alphas = alpha*.0001,
                        spy = 0,
                        players = players[setting_set],
                        osm = osm[setting_set],
                        savepath = os.path.join(os.path.join(currpath, 'checkpoint_' + str(iteration)), 'checkpoint-' + str(iteration)),
                        extended = 0,
                        teamspirit = 0
                    ))
'''
'''
with open('eval_spirit.txt', 'w+') as f:
    root = '/afs/ece.cmu.edu/usr/charlieh/ray_results/PPO/Spirit'
    for item in os.listdir(root):
        itemsplit = item.split('_')
        currpath = os.path.join(root, item)
        f.write("{blocks} {alphas:.04f} {spy} {players} {osm} {savepath} {extended} {teamspirit}\n".format(
                        blocks = 5,
                        alphas = 0.3,
                        spy = 1,
                        players = 3,
                        osm = 0,
                        savepath = os.path.join(os.path.join(currpath, 'checkpoint_407'), 'checkpoint-407'),
                        extended = 1,
                        teamspirit = float(itemsplit[2])*.01
                    ))
'''
'''
with open('eval_time.txt', 'w+') as f:
    
    for iteration in range(1, 205, 2):
        currpath = '/afs/ece.cmu.edu/usr/charlieh/ray_results/PPO/Blind/PPO_RL2_0_5_4500_0_0_2020-06-08_05-47-14ilyu34qo'
        f.write("{blocks} {alphas:.04f} {spy} {players} {osm} {savepath} {extended} {teamspirit}\n".format(
                            blocks = 5,
                            alphas = .45,
                            spy = 0,
                            players = 2,
                            osm = 0,
                            savepath = os.path.join(os.path.join(currpath, 'checkpoint_' + str(iteration)), 'checkpoint-' + str(iteration)),
                            extended = 0,
                            teamspirit = 0
                        ))
        
        currpath = '/afs/ece.cmu.edu/usr/charlieh/ray_results/PPO/Blind/PPO_RL3_0_5_3000_0_0_2020-06-08_05-47-204jxjv1eo'
        f.write("{blocks} {alphas:.04f} {spy} {players} {osm} {savepath} {extended} {teamspirit}\n".format(
                            blocks = 5,
                            alphas = .3,
                            spy = 0,
                            players = 3,
                            osm = 0,
                            savepath = os.path.join(os.path.join(currpath, 'checkpoint_' + str(iteration)), 'checkpoint-' + str(iteration)),
                            extended = 0,
                            teamspirit = 0
                        ))
        currpath = '/afs/ece.cmu.edu/usr/charlieh/ray_results/PPO/Blind/PPO_RL4_0_5_2250_0_0_2020-06-08_05-47-21wkb2998q'
        f.write("{blocks} {alphas:.04f} {spy} {players} {osm} {savepath} {extended} {teamspirit}\n".format(
                            blocks = 5,
                            alphas = .225,
                            spy = 0,
                            players = 4,
                            osm = 0,
                            savepath = os.path.join(os.path.join(currpath, 'checkpoint_' + str(iteration)), 'checkpoint-' + str(iteration)),
                            extended = 0,
                            teamspirit = 0
                        ))
'''


                    