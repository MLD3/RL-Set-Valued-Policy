#!/usr/bin/env python
# coding: utf-8

# ## LIB

# In[1]:
tol = 1e-15

# zeta_range = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
zeta_range = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12]

import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import copy
import itertools
from tqdm import tqdm
import time
import pickle

# Common policies
epsilon = 0.2

def _random_argmax(x):
    return np.random.choice(np.where(x == np.max(x))[0])

def uniformly_random_policy(Q):
    pi = np.ones_like(Q) / Q.shape[1]
    return pi

def greedy_policy(Q):
    pi = np.zeros_like(Q)
    a_star = np.array([_random_argmax(Q[s]) for s in range(len(Q))])
    for s, a in enumerate(a_star):
        pi[s, a] = 1
    return pi

def soft_greedy_policy(Q):
    pi = np.zeros_like(Q)
    for s in range(len(Q)):
        pi[s, np.where(np.isclose(Q[s], np.max(Q[s])))[0]] = 1
    return pi / pi.sum(axis=1, keepdims=True)

def epsilon_greedy_policy(Q):
    #  ε : explore
    # 1-ε: exploit
    pi = np.ones_like(Q) * epsilon / (Q.shape[1])
    a_star = np.array([_random_argmax(Q[s]) for s in range(len(Q))])
    for s, a in enumerate(a_star):
        pi[s, a] = 1 - epsilon + epsilon / (Q.shape[1])
    return pi

temperature = 1
def boltzmann_softmax_policy(Q):
    H = np.exp(Q/temperature)
    return H / np.sum(H, axis=1, keepdims=True)

# Policies induced by near-equivalent actions
zeta = None

def zeta_optimal_stochastic_policy(Q, baseline=None):
    # assigns equal probability to top near-equivalent actions
    baseline = baseline if baseline is not None else Q
    pi = np.zeros_like(Q)
    Q_cutoff = (1-zeta-tol) * baseline.max(axis=1, keepdims=True)
    pi[Q >= Q_cutoff] = 1
    pi = pi / pi.sum(axis=1, keepdims=True)
    return pi

def zeta_optimal_worst_case_policy(Q, baseline=None):
    # always choose the worst among the top near-equivalent actions
    baseline = baseline if baseline is not None else Q
    pi = np.zeros_like(Q)
    Q_cutoff = (1-zeta-tol) * baseline.max(axis=1, keepdims=True)
    pi[Q >= Q_cutoff] = 1
    a_star_hat = np.ma.masked_array(Q, 1-pi).min(axis=1, keepdims=True).data
    pi[Q != a_star_hat] = 0
    pi = pi / pi.sum(axis=1, keepdims=True)
    return pi


import matplotlib.colors as mcolors
import seaborn as sns

def plot_Q_learning_curves(Qs, Q_star=None):
    def flip(items, ncol):
        return itertools.chain(*[items[i::ncol] for i in range(ncol)])
    
    _, nS, nA = Qs.shape
    Q_history = Qs.reshape((len(Qs), -1))
    for s in range(nS):
        for a in range(nA):
            idx = (nA*s+a)
            plt.plot(Q_history[:,idx], 
                     color=(list(mcolors.TABLEAU_COLORS)*100)[s], 
                     alpha=1.0-(a//2)*0.4,
                     ls=[':', '--', '-.', '-'][a], 
                     label='$Q(s_{{{}}}, a_{{{}}})$'.format(s,a))
    plt.xlabel('num of transitions/updates')
    plt.ylabel('Q-value')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(flip(handles, nA), flip(labels, nA), bbox_to_anchor=(1.04,1), loc="upper left", ncol=nA)
    
    if Q_star is not None:
        global zeta
        V_star = Q_star.max(axis=1)
        Q_cutoff = (1 - zeta) * V_star
        for s, v in enumerate(Q_cutoff):
            plt.axhline(v, alpha=0.6, lw=0.75,
                        c=(list(mcolors.TABLEAU_COLORS)*100)[s])
    plt.show()
    return

def plot_policy(Q, Q_star=None, nS=7, S_terminal=[0,6], directions=None):
    if directions is None:
        directions = {
            0: (-1, 0),
            1: (1, 0),
            2: (-1, 0.2),
            3: (1, 0.2),
        }
    def get_direction(a):
        return directions[a]

    grid_size=(1,nS)
    nS, nA = Q.shape
    data = []
    for row in range(grid_size[0]):
        for col in range(grid_size[1]):
            s = row * grid_size[1] + col
            if s in S_terminal:
                data.append([col, row, 0, 0, 'none'])
            else:
                if Q_star is None: # plot deterministic greedy policy
                    data.append([col, row, *get_direction(Q[s].argmax()), 'k'])
                else: # plot near-optimal policy
                    try:
                        Q_cutoff = (1-zeta) * Q_star[s].max()
                        Q_tilde_s_a = Q[s][Q[s] > Q_cutoff].min()
                        for a in range(nA):
                            if Q[s,a] == Q[s].max(): # best best action
                                data.append([col, row, *get_direction(a), 'k'])
                            elif Q[s,a] == Q_tilde_s_a: # worst best action
                                data.append([col, row, *get_direction(a), 'r'])
                            elif Q[s,a] >= Q_cutoff: # Pi_a = list(np.argwhere(Q[s] > Q_cutoff)[:,0])
                                data.append([col, row, *get_direction(a), 'grey'])
                            else:
                                pass
                    except: # fall back on greedy policy
                        data.append([col, row, *get_direction(Q[s].argmax()), 'green'])
                            
    
    data = pd.DataFrame(data, columns=['x', 'y', 'u', 'v', 'color'])
    plt.figure(figsize=(5,1))
    plt.scatter(data['x'], data['y'], s=200, facecolors='none', edgecolors='k')
    plt.quiver(data['x'], data['y'], data['u'], data['v'], color=data['color'], width=0.005, scale=2, scale_units='x')
    plt.axis('off')
    plt.xlim(-0.5, nS+0.5)
    plt.ylim(-1,1)
    plt.tight_layout()
    plt.show()
    return

def print_policy(pi):
    idx = 1
    print('action  | ', end='')
    for j in range(pi.shape[1]):
        print(j+1, end=' ')
    print()
    print('-'* 18)
    for i in pi.tolist():
        print('state', idx, '|', end=' ')
        idx += 1
        for j in i:
            if j > 0:
                print(1, end=' ')
            else:
                print('-', end=' ')
        print()

def display_results_(Q):
    Q = Q[1:-1]
    V = np.max(Q, axis=1)
    print('Q')
    print(Q)
    print('V')
    print(V)
    return

def display_results(Q, Q_star=None):
    if Q_star is None:
        V = np.max(Q, axis=1)
    else:
        # always choose the worst among the top near-equivalent actions
        Q_cutoff = (1-zeta-tol) * Q_star.max(axis=1, keepdims=True)
        pi = np.zeros_like(Q)
        pi[Q >= Q_cutoff] = 1
        a_star_hat = np.ma.masked_array(Q, 1-pi).min(axis=1, keepdims=True).data
        pi[Q != a_star_hat] = 0
        pi = pi / pi.sum(axis=1, keepdims=True)
        V = (Q * pi).sum(axis=1)
    print('Q')
    print(Q)
    print('V')
    print(V)
    return Q, V

def value_iter(env, gamma=0.9, theta=1e-5):
    if not hasattr(env, 'P'):
        raise NotImplementedError
    
    P = env.P
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            old_v = V[s]
            
            ## V[s] = max {a} sum {s', r} P[s', r | s, a] * (r + gamma * V[s'])
            Q = np.zeros(env.nA)
            for a in P[s]:
                Q[a] = sum(p * (r + gamma * V[s_]) for p, s_, r, done in P[s][a])
            new_v = Q.max()
            V[s] = new_v
            delta = max(delta, np.abs(new_v - old_v))
        if delta < theta:
            break
    
    return V

def SARSA_eval(env, n_episodes, pi, gamma=1, alpha=0.1):
    global epsilon
    if not callable(alpha): # step size
        alpha_ = alpha
        alpha = lambda episode: alpha_
    
    Q = np.zeros((env.nS, env.nA))
    Gs = []
    Qs = [Q.copy()]
    for episode in tqdm(range(n_episodes)):
        G = 0
        t = 0
        np.random.seed(episode)
        env.seed(episode)
        env.reset()
        S = env.s
        done = False
        A = np.random.choice(env.nA, p=pi[S])
        while not done: # S is not a terminal state
            S_, R, done, info = env.step(A)
            A_ = np.random.choice(env.nA, p=pi[S_])
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * Q[S_,A_] - Q[S,A])
            S = S_
            A = A_
            G = G + (gamma ** t) * R
            t = t + 1
            Qs.append(Q.copy())
        Gs.append(G)
    return Q, {'Gs': np.array(Gs), 'Qs': np.array(Qs)}

def qlearn(env, n_episodes, behavior_policy, gamma=1, alpha=0.1, Q_init=None, memory=None):
    global epsilon
    if not callable(alpha): # step size
        alpha_ = alpha
        alpha = lambda episode: alpha_
    
    if Q_init is None:
        Q = np.zeros((env.nS, env.nA))
    else:
        Q = Q_init.copy().astype(float)
    
    if memory is not None:
        TD_errors = []
        Qs = [Q.copy()]
        episode = 0
        for S, A, R, S_, done in tqdm(memory):
            TD_errors.append(R + gamma * Q[S_].max())
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * Q[S_].max() - Q[S,A])
            Qs.append(Q.copy())
            if done:
                episode += 1
        return Q, {'Qs': np.array(Qs), 'TD_errors': np.array(TD_errors)}
    
    Gs = []
    Qs = [Q.copy()]
    TD_errors = []
    memory_buffer = []
    
    for episode in tqdm(range(n_episodes)):
        G = 0
        t = 0
        np.random.seed(episode)
        env.seed(episode)
        env.reset()
        S = env.s
        done = False
        while not done: # S is not a terminal state
            #p = behavior_policy(Q)[S]
            p = behavior_policy(Q[[S],:])[0]
            A = np.random.choice(env.nA, p=p)
            S_, R, done, info = env.step(A)
            memory_buffer.append((S, A, R, S_, done))
            TD_errors.append(R + gamma * Q[S_].max() - Q[S,A])
            
            # Perform update
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * Q[S_].max() - Q[S,A])
            
            S = S_
            G = G + (gamma ** t) * R
            t = t + 1
            Qs.append(Q.copy())
        Gs.append(G)
    
    return Q, {
        'Gs': np.array(Gs), # cumulative reward for each episode
        'Qs': np.array(Qs), # history of all Q-values per update
        'TD_errors': np.array(TD_errors), # temporal difference error for each update
        'memory': memory_buffer, # all trajectories/experience, tuples of (s,a,r,s', done)
    }

def expected_SARSA(env, n_episodes, behavior_policy, target_policy, gamma=1, alpha=0.1):
    global epsilon
    if not callable(alpha): # step size
        alpha_ = alpha
        alpha = lambda episode: alpha_
    
    Q = np.zeros((env.nS, env.nA))
    Gs = []
    Qs = [Q.copy()]
    for episode in tqdm(range(n_episodes)):
        G = 0
        t = 0
        np.random.seed(episode)
        env.seed(episode)
        env.reset()
        S = env.s
        done = False
        while not done: # S is not a terminal state
            A = np.random.choice(env.nA, p=behavior_policy(Q)[S])
            S_, R, done, info = env.step(A)
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * (target_policy(Q)[S_] @ Q[S_]) - Q[S,A])
            S = S_
            G = G + (gamma ** t) * R
            t = t + 1
            Qs.append(Q.copy())
        Gs.append(G)
    return Q, {'Gs': np.array(Gs), 'Qs': np.array(Qs)}

def TD_conservative(env, n_episodes, behavior_policy, target_policy=None, 
                    gamma=1, alpha=0.1, Q_init=None, Q_star=None, memory=None):
    if Q_star is None:
        assert False
    if Q_init is None:
        Q_init = np.zeros_like(Q_star)
    
    global epsilon
    if not callable(alpha): # step size
        alpha_ = alpha
        alpha = lambda episode: alpha_
    
    Q = Q_init.copy().astype(float)
    
    if memory is not None:
        TD_errors = []
        Qs = [Q.copy()]
        episode = 0
        for S, A, R, S_, done in tqdm(memory):
            TD_errors.append(R + gamma * (1-zeta) * Q_star[S_].max())
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * (1-zeta) * Q_star[S_].max() - Q[S,A])
            Qs.append(Q.copy())
            if done:
                episode += 1
        return Q, {'Qs': np.array(Qs), 'TD_errors': np.array(TD_errors)}
    
    Gs = []
    Qs = [Q.copy()]
    TD_errors = []
    for episode in tqdm(range(n_episodes)):
        G = 0
        t = 0
        np.random.seed(episode)
        env.seed(episode)
        env.reset()
        S = env.s
        done = False
        while not done: # S is not a terminal state
            A = np.random.choice(env.nA, p=behavior_policy(Q)[S])
            S_, R, done, info = env.step(A)
            TD_errors.append(R + gamma * (1-zeta) * Q_star[S_].max())
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * (1-zeta) * Q_star[S_].max() - Q[S,A])
            S = S_
            G = G + (gamma ** t) * R
            t = t + 1
            Qs.append(Q.copy())
        Gs.append(G)
    return Q, {'Gs': np.array(Gs), 'Qs': np.array(Qs), 'TD_errors': np.array(TD_errors)}

def TD_improved(env, n_episodes, behavior_policy, target_policy=None, 
                gamma=1, alpha=0.1, Q_init=None, Q_star=None, memory=None):
    if Q_star is None:
        assert False
    if Q_init is None:
        Q_init = np.zeros_like(Q_star)
    
    global epsilon
    if not callable(alpha): # step size
        alpha_ = alpha
        alpha = lambda episode: alpha_
    
    Q = Q_init.copy().astype(float)
    
    if memory is not None:
        TD_errors = []
        Qs = [Q.copy()]
        episode = 0
        for S, A, R, S_, done in tqdm(memory):
            Q_cutoff = (1-zeta) * Q_star[S_].max() # lower bound for future return
            Pi_S = np.argwhere(Q[S_] > Q_cutoff)
            if len(Pi_S) > 0: # improve the lower bound
                Q_tilde_s_a = Q[S_][Pi_S].min() # using the worst best action
            else:
                # Q_tilde_s_a = Q_cutoff # fall back to the lower bound
                Q_tilde_s_a = Q[S_].max() # fall back to the greedy action
            
            TD_errors.append(R + gamma * Q_tilde_s_a - Q[S,A])
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * Q_tilde_s_a - Q[S,A])
            Qs.append(Q.copy())
            if done:
                episode += 1
        return Q, {'Qs': np.array(Qs), 'TD_errors': np.array(TD_errors)}
    
    Gs = []
    Qs = [Q.copy()]
    TD_errors = []
    for episode in tqdm(range(n_episodes)):
        G = 0
        t = 0
        np.random.seed(episode)
        env.seed(episode)
        env.reset()
        S = env.s
        done = False
        while not done: # S is not a terminal state
            A = np.random.choice(env.nA, p=behavior_policy(Q)[S])
            S_, R, done, info = env.step(A)
            Q_cutoff = (1-zeta) * Q_star[S_].max() # lower bound for future return
            Pi_S = np.argwhere(Q[S_] > Q_cutoff)
            if len(Pi_S) > 0: # improve the lower bound
                Q_tilde_s_a = Q[S_][Pi_S].min() # using the worst best action
            else:
                # Q_tilde_s_a = Q_cutoff # fall back to the lower bound
                Q_tilde_s_a = Q[S_].max() # fall back to the greedy action
            
            TD_errors.append(R + gamma * Q_tilde_s_a - Q[S,A])
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * Q_tilde_s_a - Q[S,A])
            S = S_
            G = G + (gamma ** t) * R
            t = t + 1
            Qs.append(Q.copy())
        Gs.append(G)
    return Q, {'Gs': np.array(Gs), 'Qs': np.array(Qs), 'TD_errors': np.array(TD_errors)}


from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
from gym.envs.toy_text import discrete


# In[5]:


def visualize_policy(policy, ncol=1):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3

    _actions = ['L', 'D', 'R', 'U']
    _actions = ['←', '↓', '→', '↑']
    _actions = ['<', 'V', '>', 'A']

    for i, pi_a_s in enumerate(policy):
        # idxs = np.where(pi_a_s == pi_a_s.max())[0]
        idxs = np.where(pi_a_s > 0)[0]
        actions_str = ''.join(_actions[idx] for idx in idxs)
        print('{:^6}'.format(actions_str), end='|')
        if (i+1) % ncol == 0:
            print('\n', end='')


# In[6]:


LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

import sys
from gym import utils
class FrozenLakeEnv_slip_rand_reward(discrete.DiscreteEnv):
    def __init__(self, desc=None, map_name="4x4", slip_prob=0.0, reward_modifier=None):
        if reward_modifier is None:
            reward_modifier = [0., 0., 0., 0.]
        
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)
        
        nA = 4
        nS = nrow * ncol
        S_terminal = self.S_terminal = []

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                rand_rew = np.random.permutation(reward_modifier)
                s = to_s(row, col)
                if desc[row, col] in b'GH': # terminal absorbing states
                    S_terminal.append(s)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH': # terminal absorbing states
                        li.append((1.0, s, 0, True))
                    else:
                        for b in [(a-1)%4, a, (a+1)%4]:
                            newrow, newcol = inc(row, col, b)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G') + rand_rew[a] # reward mod determined by intended action
                            if b != a:
                                li.append((slip_prob/2.0, newstate, rew, done))
                            else:
                                li.append((1.0-slip_prob, newstate, rew, done))
        
        super().__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            with closing(outfile):
                return outfile.getvalue()


# ## Deterministic frozenlake 8x8 with intermediate rewards

np.random.seed(42)
env = FrozenLakeEnv_slip_rand_reward(
    map_name="8x8", 
    slip_prob=0.0,
    reward_modifier = [0.000, 0.001, 0.002, 0.003],
)
nS, nA = env.nS, env.nA
S_terminal = env.S_terminal

V_STAR = value_iter(env, gamma=0.9, theta=1e-20)

n_ep = 100000
gamma = 0.9
epsilon = 0.9
alpha = 0.5
# alpha_max, alpha_min, alpha_decay = 0.9, 1e-10, 1e-2
# alpha = lambda ep: alpha_min + (alpha_max-alpha_min)*np.exp(-alpha_decay*(ep//100))

# Q_star, metadata_ = qlearn(
#     env, n_ep, epsilon_greedy_policy, 
#     gamma=gamma, alpha=alpha,
# )
# with open('output/frozenlake_8x8_rrew.Q.pkl', 'wb') as f:
#     pickle.dump(metadata_, f, protocol=4)


########
# Use previous memory and Q_star
with open('output/frozenlake_8x8_rrew.Q.pkl', 'rb') as f:
    metadata_ = pickle.load(f)
    Q_star = metadata_['Qs'][-1] # Q table after the final update

print('Loaded Q_star')


# # ### Conservative

# print('Start Conservative')

# for zeta in zeta_range:
#     print('='*80)
#     print('zeta:', zeta, flush=True)
#     Q, metadata1 = TD_conservative(
#         env, n_ep, None, 
#         gamma=gamma, alpha=alpha,
#         Q_star=Q_star,
#         memory=metadata_['memory'],
#     )
#     with open('output/frozenlake_8x8_rrew.conservative.zeta={}.pkl'.format(zeta), 'wb') as f:
#         pickle.dump(metadata1, f, protocol=4)

# ### Greedy

print('Start Greedy')
for zeta in zeta_range:
    print('='*80)
    print('zeta:', zeta, flush=True)
    Q, metadata1 = TD_improved(
        env, n_ep, None, 
        gamma=gamma, alpha=alpha,
        Q_star=Q_star,
        memory=metadata_['memory'],
    )
    with open('output/frozenlake_8x8_rrew.greedy.zeta={}.pkl'.format(zeta), 'wb') as f:
        pickle.dump(metadata1, f, protocol=4)

