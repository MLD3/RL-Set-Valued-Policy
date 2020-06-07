import numpy as np
from tqdm import tqdm
import pickle
import random

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--zeta', type=float, default=0.0)
parser.add_argument('--gamma', type=float, default=0.99)
args = parser.parse_args()
zeta = args.zeta

traj_tr = pickle.load(open('trajDr_tr.pkl', 'rb'))
trajectories = traj_tr
Q_mask = np.load('action_mask.npy') # mask out actions that clinicians never taken

gamma = args.gamma or 0.99
nS, dS, nA = 750, 750, 25
n_episodes = 1000000
alpha_max, alpha_min, alpha_decay = 0.9, 1e-10, 1e-2
alpha = lambda ep: alpha_min + (alpha_max-alpha_min)*np.exp(-alpha_decay*(ep//1000))

def make_state_feature(s):
    x = np.zeros((dS))
    x[s] = 1.0
    return x


tol = 0.0
Q_star = np.load('L_qlearn_Q.npy')
Q_star = np.where(Q_mask, Q_star, np.nan) # mask out invalid actions
V_star = np.nanmax(Q_star, axis=1)

random.seed(0)
Q = np.zeros((dS, nA))
Q = Q_star.copy().astype(float)
Q = np.nan_to_num(Q) # avoid nan errors in dot product
TD_errors = []
Qs = [Q.copy()]

for episode in tqdm(range(n_episodes)):
    i = random.choice(range(len(trajectories)))
    trajectory = trajectories[i]
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
            Q_next = Q.T @ make_state_feature(S_)
            Q_cutoff = min(V_star[S_], (1-zeta-tol) * V_star[S_]) # lower bound for future return, handles negative value
            Pi_S = np.argwhere(
                np.where(Q_mask[S_], Q_next, -np.inf) > Q_cutoff
            )
            if len(Pi_S) > 0: # improve the lower bound
                assert not np.isnan(Q_next[Pi_S]).all()
                V_next = Q_next[Pi_S].min() # using the worst best action
            else:
                V_next = np.max(Q_next[Q_mask[S_]]) # fall back to the greedy action
            target = R + gamma * V_next
        
        Sx = make_state_feature(S)
        TD_err = target - (Q.T @ Sx)[A]
        Q[:,A] = Q[:,A] + alpha(episode) * TD_err * Sx
        TD_errors.append(TD_err)
    
    if episode % 500 == 0:
        Qs.append(Q.copy())

metadata = {
    'Qs': Qs, 
    'TD_errors': TD_errors,
}

np.save('output/L_svp_near-greedy_Q_gamma={}_zeta={}.npy'.format(gamma, zeta), Q)
pickle.dump(metadata, open('output/L_svp_near-greedy_metadata_gamma={}_zeta={}.p'.format(gamma, zeta), 'wb'))

exit()
