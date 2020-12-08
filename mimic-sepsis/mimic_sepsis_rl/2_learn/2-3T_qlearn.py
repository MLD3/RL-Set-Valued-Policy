import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from collections import defaultdict
import pickle
import itertools
import random

random.seed(0)
traj_tr = pickle.load(open('trajDr_tr.pkl', 'rb'))

# mask out actions that clinicians never taken
Q_mask = np.load('action_mask.npy')

gamma = 0.99
nS, nA = 750, 25
n_episodes = 1000000
alpha_max, alpha_min, alpha_decay = 0.9, 1e-10, 1e-2
alpha = lambda ep: alpha_min + (alpha_max-alpha_min)*np.exp(-alpha_decay*(ep//1000))

def run_qlearn(trajectories, n_episodes, gamma, alpha=0.5, Q_init=None):
    if not callable(alpha): # step size
        alpha_ = alpha
        alpha = lambda episode: alpha_
    
    if Q_init is None:
        Q = np.zeros((nS, nA))
    else:
        Q = Q_init.copy().astype(float)
    
    Q[~Q_mask] = np.nan
    
    TD_errors = []
    Qs = [Q.copy()]
    for episode in tqdm(range(n_episodes)):
        trajectory = random.choice(trajectories)
        for t, transition in enumerate(trajectory):
            S = transition['s']
            A = transition['a']
            R = transition['r']
            S_ = transition['s_']
            done = transition['done']
            if done:
                target = R
            else:
                assert S_ is not None
                target = R + gamma * np.nanmax(Q[S_])

            TD_errors.append(target - Q[S,A])
            Q[S,A] = Q[S,A] + alpha(episode) * (target - Q[S,A])
        
        if episode % 500 == 0:
            Qs.append(Q.copy())
    
    return Q, {
        'Qs': Qs, 
        'TD_errors': TD_errors,
    }

Q, metadata = run_qlearn(traj_tr, n_episodes, gamma, alpha)
np.save('qlearn_Q.npy', Q)
pickle.dump(metadata, open('qlearn_metadata.p', 'wb'))

exit()
