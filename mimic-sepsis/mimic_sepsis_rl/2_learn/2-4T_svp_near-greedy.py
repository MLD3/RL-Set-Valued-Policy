import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from collections import defaultdict
import pickle
import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--zeta', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.99)
args = parser.parse_args()
zeta = args.zeta

random.seed(0)
traj_tr = pickle.load(open('trajDr_tr.pkl', 'rb'))

# mask out actions that clinicians never taken
Q_mask = np.load('action_mask.npy')

gamma = args.gamma or 0.99
nS, nA = 750, 25
n_episodes = 1000000
alpha_max, alpha_min, alpha_decay = 0.9, 1e-10, 1e-2
alpha = lambda ep: alpha_min + (alpha_max-alpha_min)*np.exp(-alpha_decay*(ep//1000))

tol = 0.0
Q_star = np.load('qlearn_Q.npy')
V_star = np.nanmax(Q_star, axis=1)

def run_near_greedy(trajectories, n_episodes, gamma, alpha=0.5, Q_init=None):
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
                Q_cutoff = min(V_star[S_], (1-zeta-tol) * V_star[S_]) # lower bound for future return
                Pi_S = np.argwhere(
                    np.where(Q_mask[S_], Q[S_], -np.inf) > Q_cutoff
                )
                if len(Pi_S) > 0: # improve the lower bound
                    assert not np.isnan(Q[S_][Pi_S]).all()
                    Q_tilde_s_a = Q[S_][Pi_S].min() # using the worst best action
                else:
                    Q_tilde_s_a = np.nanmax(Q[S_]) # fall back to the greedy action
                target = R + gamma * Q_tilde_s_a
            
            TD_errors.append(target - Q[S,A])
            Q[S,A] = Q[S,A] + alpha(episode) * (target - Q[S,A])
        
        if episode % 500 == 0:
            Qs.append(Q.copy())
    
    return Q, {
        'Qs': Qs, 
        'TD_errors': TD_errors,
    }

Q, metadata = run_near_greedy(traj_tr, n_episodes, gamma, alpha, Q_init=Q_star)
np.save('output_ql/svp_near-greedy_Q_gamma={}_zeta={}.npy'.format(gamma, zeta), Q)
pickle.dump(metadata, open('output_ql/svp_near-greedy_metadata_gamma={}_zeta={}.p'.format(gamma, zeta), 'wb'))

exit()
