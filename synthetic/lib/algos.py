import gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import copy
import itertools
from tqdm import tqdm
import time
import pickle
from joblib import Parallel, delayed

def policy_eval(env, pi, gamma, theta=1e-10):
    if not hasattr(env, 'P'):
        raise NotImplementedError
    
    P = env.P
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            old_v = V[s]
            
            ## V[s] = sum {a} policy[s|a] sum {s', r} P[s', r | s, a] * (r + gamma * V[s'])
            new_v = 0
            for a in P[s]:
                new_v += pi[s, a] * sum(p * (r + gamma * V[s_] * (1-done)) for p, s_, r, done in P[s][a])
            V[s] = new_v
            delta = max(delta, np.abs(new_v - old_v))
        if delta < theta:
            break
    
    return V

def policy_eval_worst_case(env, pi, gamma, theta=1e-10):
    if not hasattr(env, 'P'):
        raise NotImplementedError
    
    P = env.P
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            old_v = V[s]
            
            ## V[s] = min {a \in π(s)} policy[s|a] sum {s', r} P[s', r | s, a] * (r + gamma * V[s'])
            Q_s = []
            for a in P[s]:
                if pi[s][a] > 0:
                    Q_s.append(sum(p * (r + gamma * V[s_] * (1-done)) for p, s_, r, done in P[s][a]))
            new_v = np.min(Q_s)
            V[s] = new_v
            delta = max(delta, np.abs(new_v - old_v))
        if delta < theta:
            break
    
    return V

def value_iter(env, gamma, theta=1e-10):
    if not hasattr(env, 'P'):
        raise NotImplementedError
    
    P = env.P
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            old_v = V[s]
            
            ## V[s] = max {a} sum {s', r} P[s', r | s, a] * (r + gamma * V[s'])
            Q_s = np.zeros(env.nA)
            for a in P[s]:
                Q_s[a] = sum(p * (r + gamma * V[s_] * (1-done)) for p, s_, r, done in P[s][a])
            new_v = Q_s.max()
            V[s] = new_v
            delta = max(delta, np.abs(new_v - old_v))
        if delta < theta:
            break
    
    return V

def value_iter_near_greedy(env, gamma, zeta, V_star=None, theta=1e-10, max_iter=1000):
    if not hasattr(env, 'P'):
        raise NotImplementedError
    if V_star is None:
        assert False
    
    P = env.P
    V = V_star.copy().astype(float)
    policies = []
    n_iter = 0
    
    # save SVP after each iteration
    Q = V2Q(env, V, gamma)
    Pi = np.zeros_like(Q)
    for s in range(env.nS):
        Pi[s] = (Q[s] > (1-zeta) * V_star[s])
    policies.append(Pi)

    while True:
        delta = 0.0
        for s in range(env.nS):
            old_v = V[s]
            
            Q_s = np.zeros(env.nA)
            for a in P[s]:
                Q_s[a] = sum(p * (r + gamma * V[s_]) for p, s_, r, done in P[s][a])
            
            Q_cutoff = (1-zeta) * V_star[s] # lower bound for return
            Pi_S = np.argwhere(Q_s > Q_cutoff)
            if len(Pi_S) > 0: # improve the lower bound
                new_v = Q_s[Pi_S].min() # using the worst best action
            else:
                new_v = Q_s.max() # fall back to the greedy action
            
            V[s] = new_v
            delta = max(delta, np.abs(new_v - old_v))
        
        # save SVP after each iteration
        Q = V2Q(env, V, gamma)
        Pi = np.zeros_like(Q)
        for s in range(env.nS):
            Pi[s] = (Q[s] > (1-zeta) * V_star[s])
        policies.append(Pi)
        
        n_iter += 1
        if ((policies[-1] == policies[-2]).all() and delta < theta) or n_iter >= max_iter:
            break
    
    return V, {
        'policies': policies,
        'delta': delta,
    }

def value_iter_near_greedy_prob(env, gamma, zeta, V_star=None, rho=0.1, theta=1e-10, max_iter=1000):
    if not hasattr(env, 'P'):
        raise NotImplementedError
    if V_star is None:
        assert False
    
    P = env.P
    V = V_star.copy().astype(float)
    policies = []
    n_iter = 0
    
    # save SVP after each iteration
    Q = V2Q(env, V, gamma)
    Pi = np.zeros_like(Q)
    for s in range(env.nS):
        Pi[s] = (Q[s] > (1-zeta) * V_star[s])
    policies.append(Pi)

    while True:
        delta = 0.0
        for s in range(env.nS):
            old_v = V[s]
            
            Q_s = np.zeros(env.nA)
            for a in P[s]:
                Q_s[a] = sum(p * (r + gamma * V[s_]) for p, s_, r, done in P[s][a])
            
            Pi_S = np.argwhere(Q_s > (1-zeta) * V_star[s]) # lower bound for return
            if len(Pi_S) > 0: # improve the lower bound
                new_v = (
                      # max
                      (1-rho) * Q_s.max() +
                      # other actions in pi
                      rho * np.mean(Q_s[Pi_S])
                )
            else:
                new_v = Q_s.max() # fall back to the greedy action
            
            V[s] = new_v
            delta = max(delta, np.abs(new_v - old_v))
        
        # save SVP after each iteration
        Q = V2Q(env, V, gamma)
        Pi = np.zeros_like(Q)
        for s in range(env.nS):
            Pi[s] = (Q[s] > (1-zeta) * V_star[s])
        policies.append(Pi)
        
        n_iter += 1
        if delta < theta or n_iter >= max_iter:
            break
    
    return V, {
        'policies': policies,
        'delta': delta,
    }


def qlearn(
    env, n_episodes, behavior_policy, gamma, alpha=0.1, epsilon=1.0, 
    Q_init=None, memory=None, save_Q=0,
):
    if not callable(epsilon):
        epsilon_func = lambda episode: epsilon
    else:
        epsilon_func = epsilon
    
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
            TD_errors.append(R + gamma * Q[S_].max() - Q[S,A])
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * Q[S_].max() - Q[S,A])
            if save_Q: Qs.append(Q.copy())
            if done:
                episode += 1
        return Q, {
            'Qs': np.array(Qs), 
            'TD_errors': np.array(TD_errors),
            'memory': memory,
        }
    
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
            p = behavior_policy(Q[[S],:], dict(epsilon=epsilon_func(episode)))[0]
            A = np.random.choice(env.nA, p=p)
            S_, R, done, info = env.step(A)
            memory_buffer.append((S, A, R, S_, done))
            TD_errors.append(R + gamma * Q[S_].max() - Q[S,A])
            
            # Perform update
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * Q[S_].max() - Q[S,A])
            
            S = S_
            G = G + (gamma ** t) * R
            t = t + 1
            if save_Q: Qs.append(Q.copy())
        Gs.append(G)
    
    return Q, {
        'Gs': np.array(Gs), # cumulative reward for each episode
        'Qs': np.array(Qs), # history of all Q-values per update
        'TD_errors': np.array(TD_errors), # temporal difference error for each update
        'memory': memory_buffer, # all trajectories/experience, tuples of (s,a,r,s', done)
    }

def TD_near_greedy(
    env, n_episodes, behavior_policy,
    gamma, zeta, epsilon=1.0, alpha=0.1, Q_init=None, Q_star=None, memory=None,
    save_Q=0,
):
    if Q_star is None:
        assert False
    if Q_init is None:
        Q_init = np.zeros_like(Q_star)
    
    if not callable(epsilon):
        epsilon_func = lambda episode: epsilon
    else:
        epsilon_func = epsilon
    
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
            if save_Q: Qs.append(Q.copy())
            if done:
                episode += 1
        return Q, {
            'Qs': np.array(Qs), 
            'TD_errors': np.array(TD_errors),
            'memory': memory,
        }
    
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
            p = behavior_policy(Q[[S],:], dict(epsilon=epsilon_func(episode)))[0]
            A = np.random.choice(env.nA, p=p)
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
            if save_Q: Qs.append(Q.copy())
        Gs.append(G)
    return Q, {
        'Gs': np.array(Gs), 
        'Qs': np.array(Qs), 
        'TD_errors': np.array(TD_errors)
    }

########

def V2Q(env, V, gamma):
    Q = np.zeros((env.nS, env.nA))
    for s in range(env.nS):
        for a in env.P[s]:
            Q[s,a] = sum(p * (r + 
                            (0 if done else gamma * V[s_])
                           ) for p, s_, r, done in env.P[s][a])
    return Q

def _random_argmax(x):
    return np.random.choice(np.where(x == np.max(x))[0])

def uniformly_random_policy(Q, args):
    pi = np.ones_like(Q) / Q.shape[1]
    return pi

def greedy_policy(Q, args):
    pi = np.zeros_like(Q)
    a_star = np.array([_random_argmax(Q[s]) for s in range(len(Q))])
    for s, a in enumerate(a_star):
        pi[s, a] = 1
    return pi

def soft_greedy_policy(Q, args):
    pi = np.zeros_like(Q)
    for s in range(len(Q)):
        pi[s, np.where(np.isclose(Q[s], np.max(Q[s])))[0]] = 1
    return pi / pi.sum(axis=1, keepdims=True)

def epsilon_greedy_policy(Q, args):
    #  ε : explore
    # 1-ε: exploit
    epsilon = args['epsilon']
    pi = np.ones_like(Q) * epsilon / (Q.shape[1])
    a_star = np.array([_random_argmax(Q[s]) for s in range(len(Q))])
    for s, a in enumerate(a_star):
        pi[s, a] = 1 - epsilon + epsilon / (Q.shape[1])
    return pi

def zeta_optimal_SVP(Q, args):
    # assigns 1 to top near-equivalent actions
    Q_star = args['Q_star']
    zeta = args['zeta']
    tol = args.get('tol', 1e-10)
    pi = np.zeros_like(Q)
    Q_cutoff = (1-zeta-tol) * Q_star.max(axis=1, keepdims=True)
    pi[Q >= Q_cutoff] = 1
    return pi.astype(int)

def zeta_optimal_stochastic_policy(Q, args):
    # assigns equal probability to top near-equivalent actions
    Q_star = args['Q_star']
    zeta = args['zeta']
    tol = args.get('tol', 1e-10)
    pi = np.zeros_like(Q)
    Q_cutoff = (1-zeta-tol) * Q_star.max(axis=1, keepdims=True)
    pi[Q >= Q_cutoff] = 1
    pi = pi / pi.sum(axis=1, keepdims=True)
    return pi

def zeta_optimal_worst_case_policy(Q, args):
    # always choose the worst among the top near-equivalent actions
    Q_star = args['Q_star']
    zeta = args['zeta']
    tol = args.get('tol', 1e-10)
    pi = np.zeros_like(Q)
    Q_cutoff = (1-zeta-tol) * Q_star.max(axis=1, keepdims=True)
    pi[Q >= Q_cutoff] = 1
    a_star_hat = np.ma.masked_array(Q, 1-pi).min(axis=1, keepdims=True).data
    pi[Q != a_star_hat] = 0
    pi = pi / pi.sum(axis=1, keepdims=True)
    return pi

###############

def TD_conservative(
    env, n_episodes, behavior_policy,
    gamma, zeta, epsilon=1.0, alpha=0.1, Q_init=None, Q_star=None, memory=None,
    save_Q=0,
):
    if Q_star is None:
        assert False
    if Q_init is None:
        Q_init = np.zeros_like(Q_star)
    
    if not callable(epsilon):
        epsilon_func = lambda episode: epsilon
    else:
        epsilon_func = epsilon
    
    if not callable(alpha): # step size
        alpha_ = alpha
        alpha = lambda episode: alpha_
    
    Q = Q_init.copy().astype(float)
    
    if memory is not None:
        TD_errors = []
        Qs = [Q.copy()]
        episode = 0
        for S, A, R, S_, done in tqdm(memory):
            TD_errors.append(R + gamma * (1-zeta) * Q_star[S_].max() - Q[S,A])
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * (1-zeta) * Q_star[S_].max() - Q[S,A])
            if save_Q: Qs.append(Q.copy())
            if done:
                episode += 1
        return Q, {
            'Qs': np.array(Qs), 
            'TD_errors': np.array(TD_errors),
            'memory': memory,
        }
    
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
            if save_Q: Qs.append(Q.copy())
        Gs.append(G)
    return Q, {'Gs': np.array(Gs), 'Qs': np.array(Qs), 'TD_errors': np.array(TD_errors)}

def TD_near_greedy_baseline(
    env, n_episodes, behavior_policy,
    gamma, zeta, epsilon=1.0, alpha=0.1, Q_init=None, Q_star=None, memory=None,
    save_Q=0,
):
    if Q_star is not None:
        assert False
    if Q_init is None:
        Q_init = np.zeros((env.nS, env.nA))
    
    if not callable(epsilon):
        epsilon_func = lambda episode: epsilon
    else:
        epsilon_func = epsilon
    
    if not callable(alpha): # step size
        alpha_ = alpha
        alpha = lambda episode: alpha_
    
    Q = Q_init.copy().astype(float)
    
    if memory is not None:
        TD_errors = []
        Qs = [Q.copy()]
        episode = 0
        for S, A, R, S_, done in tqdm(memory):
            Q_cutoff = (1-zeta) * Q[S_].max() # lower bound for future return
            Pi_S = np.argwhere(Q[S_] > Q_cutoff)
            if len(Pi_S) > 0: # improve the lower bound
                Q_tilde_s_a = Q[S_][Pi_S].min() # using the worst best action
            else:
                # Q_tilde_s_a = Q_cutoff # fall back to the lower bound
                Q_tilde_s_a = Q[S_].max() # fall back to the greedy action
            
            TD_errors.append(R + gamma * Q_tilde_s_a - Q[S,A])
            Q[S,A] = Q[S,A] + alpha(episode) * (R + gamma * Q_tilde_s_a - Q[S,A])
            if save_Q: Qs.append(Q.copy())
            if done:
                episode += 1
        return Q, {
            'Qs': np.array(Qs), 
            'TD_errors': np.array(TD_errors),
            'memory': memory,
        }
    
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
            p = behavior_policy(Q[[S],:], dict(epsilon=epsilon_func(episode)))[0]
            A = np.random.choice(env.nA, p=p)
            S_, R, done, info = env.step(A)
            Q_cutoff = (1-zeta) * Q[S_].max() # lower bound for future return
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
            if save_Q: Qs.append(Q.copy())
        Gs.append(G)
    return Q, {
        'Gs': np.array(Gs), 
        'Qs': np.array(Qs), 
        'TD_errors': np.array(TD_errors)
    }

def SARSA_eval(env, n_episodes, pi, gamma, alpha=0.1):
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
    return Q, {
        'Gs': np.array(Gs), 
        'Qs': np.array(Qs)
    }

def MC_eval(env, n_episodes, pi, gamma=1, max_steps=1000, parallel=True):
    def _MC_generate_episode(env, pi, seed):
        np.random.seed(seed)
        env.seed(seed)
        env.reset()

        # Generate an episode
        S = env.s = seed % env.nS
        done = False
        t = 0
        transitions = []
        while not done and t < max_steps: # S is not a terminal state
            A = np.random.choice(env.nA, p=pi[S])
            S_, R, done, info = env.step(A)
            transitions.append((S,A,R,S_))
            S = S_
            t = t + 1

        # Calculate return for each visited state
        trajectory = []
        G = 0
        for t in reversed(range(len(transitions))):
            S, A, R, S_ = transitions[t]
            G = gamma * G + R
            trajectory.append((S,A,R,S_,G))

        return list(reversed(trajectory))
    
    if parallel:
        out = Parallel(n_jobs=64)(delayed(_MC_generate_episode)(env, pi, int(1e5+i)) for i in range(n_episodes))
    else:
        out = [_MC_generate_episode(env, pi, int(1e5+i)) for i in range(n_episodes)]
    
    all_transitions = sum(out, [])
    returns_by_state = {
        s: [G for (S,A,R,S_,G) in all_transitions if s == S]
        for s in range(env.nS)
    }
    V_pi = np.array([np.mean(Gs) for s, Gs in sorted(returns_by_state.items())])
    return V_pi

def expected_SARSA(env, n_episodes, behavior_policy, target_policy, gamma, alpha=0.1):
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
    return Q, {
        'Gs': np.array(Gs), 
        'Qs': np.array(Qs)
    }
