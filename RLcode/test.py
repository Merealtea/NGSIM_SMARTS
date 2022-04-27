from types import prepare_class
import torch
import numpy as np
#import tqdm
import pickle
#from random import choice
from traffic_simulator import TrafficSim
import os
from modelFCN import Policynet

def data_loader(path):
    files = os.listdir(path)
    total_state = []
    total_actions = []
    total_next_state = []
    total_terminal = []

    for f in files:
        with open(path + '/{}'.format(f) ,'rb') as t:
            expert = pickle.load(t)
    
        exp_state = expert['observations']
        exp_actions = expert['actions']
        exp_terminal = expert['terminals']
        exp_next_state = expert['next_observations']
    
        exp_state = np.concatenate(list(map(lambda x:np.array(x).squeeze(1),exp_state)))
        exp_actions = np.concatenate(exp_actions)
        exp_terminal = np.concatenate(exp_terminal)
        exp_next_state = np.concatenate(list(map(lambda x:np.array(x).squeeze(1),exp_next_state)))

        total_state.append(exp_state)
        total_actions.append(exp_actions)
        total_next_state.append(exp_next_state)
        total_terminal.append(exp_terminal)
        break
    total_state = np.concatenate(total_state)
    total_actions = np.concatenate(total_actions)
    total_next_state = np.concatenate(total_next_state)
    total_terminal = np.concatenate(total_terminal)
    print(total_actions.shape)

    return total_state,total_actions,total_next_state,total_terminal

if __name__ == "__main__":
    
    #data_loader('./expert')
    # gen = Policynet(40,2)
    # gen.eval()

    # t1 = []
    # t2 = []
    # t3 = []


    # for i in range(4):
    #     state = torch.FloatTensor(np.ones((1,40)))
    #     policy = gen(state)
    #     act = policy.sample()
    #     t1.append(state)
    #     t2.append(policy)
    #     t3.append(act)
    # t1 = torch.cat(t1)
    # t2 = torch.stack(t2)
    # t3 = torch.cat(t3)

    # p = t2.log_prob(t3)
    x = np.array([5,5])
    x = x if x <2.5 else 2.5
    print(x)