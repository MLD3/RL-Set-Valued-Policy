import gym
from gym.envs.toy_text import discrete
import numpy as np

class MDP_Env(discrete.DiscreteEnv):
    def __init__(self, p_transition, p_reward, S_terminal, isd=None):
        nS, nA, _ = p_transition.shape
        if isd is None:
            isd = np.ones(nS)
            isd = isd / isd.sum()
        
        P = {s : {a : [] for a in range(nA)} for s in range(nS)}
        for s in range(nS):
            for a in range(nA):
                for s_ in range(nS):
                    if p_transition[s,a,s_] > 0:
                        P[s][a].append((p_transition[s,a,s_], s_, p_reward[s,a], s_ in S_terminal))
        super().__init__(nS, nA, P, isd)

class Chain(MDP_Env):
    def __init__(self, nS=7, nA=4, r_modifier=np.arange(0, 0.05, 0.01), r_replace=True):
        self.nS, self.nA = nS, nA
        S_terminal = self.S_terminal = [nS-1]
        p_transition = np.zeros((nS, nA, nS))
        for s in range(nS):
            for a in range(nA):
                if s in S_terminal:
                    p_transition[s,a,s] = 1
                else:
                    p_transition[s,a,min(s+1, nS-1)] = 1

        p_reward = np.zeros((nS, nA))
        p_reward[nS-2, :] = 1
        for s in range(nS):
            if s not in S_terminal:
                p_reward[s,:] += np.random.choice(r_modifier, (nA), replace=r_replace)
        
        super().__init__(p_transition, p_reward, S_terminal)


class ChainWithCycles(MDP_Env):
    def __init__(self, nS=7, nA=4, r_modifier=np.arange(0, 0.05, 0.01), r_replace=True):
        self.nS, self.nA = nS, nA
        S_terminal = self.S_terminal = [nS-1]
        p_transition = np.zeros((nS, nA, nS))
        for s in range(nS):
            for a in range(nA):
                if s in S_terminal:
                    p_transition[s,a,s] = 1
                else:
                    if a % 2 == 0:
                        p_transition[s,a,max(s-1, 0)] = 1
                    else:
                        p_transition[s,a,min(s+1, nS-1)] = 1

        p_reward = np.zeros((nS, nA))
        p_reward[nS-2, 1] = 1
        p_reward[nS-2, 3] = 1
        for s in range(nS):
            if s not in S_terminal:
#                 p_reward[s,3] += r_modifier
                p_reward[s,:] += np.random.choice(r_modifier, (nA), replace=r_replace)

        super().__init__(p_transition, p_reward, S_terminal)

class RandomDAG(MDP_Env):
    def __init__(self, nS=5, nA=4):
        S_terminal = self.S_terminal = [nS-1]
        p_transition = np.zeros((nS, nA, nS))
        for s in range(nS):
            if s in S_terminal:
                p_transition[s, :, s] = 1
            else:
                for a in range(nA):
                    s_ = np.random.choice(range(s+1, nS))
                    p_transition[s, a, s_] = 1

        isd = np.zeros(nS)
        isd[0] = 1

        p_reward = np.random.choice(np.arange(0, 1, 0.1), (nS, nA))
        p_reward[nS-1, :] = 0
        p_reward[np.random.choice(nS-1), np.random.choice(nA)] = 5
        
        super().__init__(p_transition, p_reward, S_terminal)

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
