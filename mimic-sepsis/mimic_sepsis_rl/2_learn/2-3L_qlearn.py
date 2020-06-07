import numpy as np
from tqdm import tqdm
import pickle
import random

traj_tr = pickle.load(open('trajDr_tr.pkl', 'rb'))
trajectories = traj_tr
Q_mask = np.load('action_mask.npy') # mask out actions that clinicians never taken

gamma = 0.99
nS, dS, nA = 750, 750, 25
n_episodes = 1000000
alpha_max, alpha_min, alpha_decay = 0.9, 1e-10, 1e-2
alpha = lambda ep: alpha_min + (alpha_max-alpha_min)*np.exp(-alpha_decay*(ep//1000))

def make_state_feature(s):
    x = np.zeros((dS))
    x[s] = 1.0
    return x


random.seed(0)
Q = np.zeros((dS, nA))
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
            target = R + gamma * np.max(Q_next[Q_mask[S_]])
        
        Sx = make_state_feature(S)
        TD_err = target - (Q.T @ Sx)[A]
        Q[:,A] = Q[:,A] + alpha(episode) * TD_err * Sx
        TD_errors.append(TD_err)
    
    if episode % 500 == 0:
        Qs.append(Q)

metadata = {
    'Qs': Qs, 
    'TD_errors': TD_errors,
}
np.save('L_qlearn_Q.npy', Q)
pickle.dump(metadata, open('L_qlearn_metadata.p', 'wb'))
