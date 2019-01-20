import numpy as np
import torch
import torch.nn as nn
import wrappers
import dqn
import argparse
import time
import matplotlib.pyplot as plt
ENV = 'PongNoFrameskip-v4'

class Agent:

    def __init__(self, env):
        self.env = env
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
    
    def play_step(self, net, device='cpu'):
        state_a = np.array([self.state], copy=False)
        state_v = torch.tensor(state_a).to(device)
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        action = int(act_v.item())
        new_state, reward, is_done, _ = self.env.step(action)
        self.env.render()
        time.sleep(0.01)
        self.total_reward += reward
        self.state = new_state
        if is_done:
            self._reset()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True, action='store_true', help='Enable cuda')
    parser.add_argument("--env", default=ENV, help='Name of the environment, default = ' + ENV)
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')
    env = wrappers.make_env(args.env)
    net = dqn.DQN(env.observation_space.shape, env.action_space.n).to(device)
    net.load_state_dict(torch.load('PongNoFrameskip-v4-best.dat'))
    agent = Agent(env)
    while True:
       agent.play_step(net, device)

if __name__ == '__main__':
    main()