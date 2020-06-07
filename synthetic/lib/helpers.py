import matplotlib.colors as mcolors
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import numpy as np

def np_exclude(x, exclude):
    return x[~np.isin(np.arange(len(x)), exclude)]

def frozenlake_visualize_policy(policy, ncol=1):
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


def plot_Q_learning_curves(Qs, Q_star=None, zeta=0.0):
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
        V_star = Q_star.max(axis=1)
        Q_cutoff = (1 - zeta) * V_star
        for s, v in enumerate(Q_cutoff):
            plt.axhline(v, alpha=0.6, lw=0.75,
                        c=(list(mcolors.TABLEAU_COLORS)*100)[s])
    plt.show()
    return

def plot_policy_old(Q, Q_star=None, zeta=None, nS=7, S_terminal=[0,6], directions=None):
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

def calculate_results(Q, Q_star=None, zeta=0.0, tol=1e-5):
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
    return Q, V

def display_results(Q, Q_star=None, zeta=0.0, tol=1e-5):
    Q, V = calculate_results(Q, Q_star, zeta, tol)
    print('Q')
    print(Q)
    print('V')
    print(V)
    return Q, V
