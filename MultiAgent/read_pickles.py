import numpy as np
reslist = np.load('/afs/ece.cmu.edu/usr/charlieh/eval_results/RLvsOSM_0_5_4000_0_0.npy', allow_pickle=True)
for res in reslist:
    print(res)