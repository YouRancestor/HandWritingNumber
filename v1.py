from numpy.core.fromnumeric import argmax
import torch
from cv2 import cv2
from torch.nn.modules.linear import Linear
import torchvision
from torch.utils import data
from torchvision import transforms
from torch import nn
from torch import optim

import numpy as np
import gym
from gym import spaces

class Indicator(gym.Env):
    def __init__(self) -> None:
        super().__init__()
        # Action space:
        # 1) Arrorw keys: Discrete 3 - Noop[0], Left[1], Right[2] - params: min: 0, max: 2
        # 2) Button: Discrete 2 - NOOP[0], Pressed[1] - params: min: 0, max: 1
        self.action_space = spaces.MultiDiscrete([3, 2])
        self.observation_space = spaces.MultiBinary([3, 10])
        self.reset()

    def reset(self):
        self.state_truth = np.random.randint(0, 10)
        self.state_finger = np.random.randint(0, 10)
        self.state_button = -1 # -1表示没按下，0~9表示已按下对应位置。

        state = np.zeros((3, 10))
        state[0][self.state_truth] = 1
        state[1][self.state_finger] = 1
        return state

    def move(self, action_move):
        if action_move == 0:
            return
        if action_move == 1:
            self.state_finger -= 1
            if self.state_finger < 0:
                self.state_finger += 10
        if action_move == 2:
            self.state_finger += 1
            if self.state_finger > 9:
                self.state_finger -= 10

    def button(self, action_button):
        if (action_button == 0) == (self.state_button == -1):
            # 跟上一时刻状态相同
            return
        else:
            if self.state_button == -1:
                # 上一时刻未按下
                # 按下finger对应位置
                self.state_button = self.state_finger
            else:
                # 上一时刻已按下
                # 抬起
                self.state_button = -1
            
    
    def step(self, action):
        action_move = action[0]
        action_button = action[1]
        
        self.move(action_move)
        self.button(action_button)

        state = np.zeros((3, 10))
        state[1][self.state_finger] = 1
        if self.state_button == -1:
            pass
        else:
            state[2][self.state_button] = 1
        
        reward = 0
        if self.state_button == self.state_truth:
            reward += 1
            old_truth = self.state_truth
            while self.state_truth == old_truth:
                self.state_truth = np.random.randint(0, 10)
        elif self.state_button != -1:
            reward -= 1
        state[0][self.state_truth] = 1

        return state, reward, False, {}

class Agent():
    def __init__(self, input_dim, internal_feature_dim, action_dim) -> None:
        '''
        input_dim: 输入observation的维数
        internal_feature_dim: 内部表示空间的维数
        action_dim: 输出action空间的维数
        '''
        super().__init__()
        self.net_observe = nn.Linear(input_dim, internal_feature_dim)
        self.net_association = nn.LSTM(internal_feature_dim + action_dim, internal_feature_dim)
        self.net_internal_to_observation = nn.Linear(internal_feature_dim, input_dim)
        self.net_action = nn.Linear(internal_feature_dim*2, action_dim)
        self.net_intention_updating = nn.LSTM(internal_feature_dim*2+action_dim, internal_feature_dim)
        self.net_predict = nn.Linear(internal_feature_dim + action_dim, internal_feature_dim)
        self.loss_fn_mse = nn.MSELoss()
        self.loss_fn_cross_entropy = nn.CrossEntropyLoss()

        all_params = []
        all_params.extend(self.net_action.parameters())
        all_params.extend(self.net_association.parameters())
        all_params.extend(self.net_intention_updating.parameters())
        all_params.extend(self.net_internal_to_observation.parameters())
        # all_params.extend(self.net_observe.parameters())
        self.param_collector = optim.Adam(all_params)

        pred_params = []
        pred_params.extend(self.net_observe.parameters())
        pred_params.extend(self.net_predict.parameters())
        pred_params.extend(self.net_association.parameters())
        self.pred_updater = optim.Adam(pred_params)

        label_params = []
        label_params.extend(self.net_internal_to_observation.parameters())
        label_params.extend(self.net_intention_updating.parameters())
        self.label_updater = optim.Adam(label_params)

        pred_mse_params = []
        pred_mse_params.extend(self.net_intention_updating.parameters())
        pred_mse_params.extend(self.net_action.parameters())
        self.pred_mse_updater = optim.Adam(pred_mse_params)

        self.pred_X = torch.zeros((internal_feature_dim))
        self.pred_ob = None
        self.association = None
        self.last_action = torch.zeros((action_dim))
        self.intention = torch.zeros((internal_feature_dim))

        # LSTM state init
        self.association_state = (torch.zeros((1,1,internal_feature_dim)), torch.zeros((1,1,internal_feature_dim)))
        self.intention_updating_state = (torch.zeros((1,1,internal_feature_dim)), torch.zeros((1,1,internal_feature_dim)))

    def observe(self, input):
        X = self.net_observe(torch.flatten(input))
        return X

    def gen_assotiation(self, X):
        '''
        产生联想
        '''
        concat_input = torch.cat([X, self.last_action]).reshape((1,1,-1))
        assotiation, self.association_state = self.net_association(concat_input, self.association_state)
        return assotiation

    def internal_to_observation(self, intention):
        expecting = self.net_internal_to_observation(intention)
        return expecting

    def update_intention(self, X, pred_X, last_action):
        concat_input = torch.cat([X, pred_X, last_action]).reshape(1,1,-1)
        intention, self.intention_updating_state = self.net_intention_updating(concat_input, self.intention_updating_state)
        return intention

    def take_action(self, X, intention):
        action = self.net_action(torch.cat([X, torch.flatten(intention)]))
        move = torch.softmax(action[0:3], -1)
        button = torch.softmax(action[3:5], -1)
        action = torch.cat([move, button])
        return action

    def pred(self, X, action):
        pred_X = self.net_predict(torch.cat([X, action]))
        return pred_X

    def set_label(self, label):
        loss = self.loss_fn_mse(torch.flatten(self.expecting), torch.flatten(label))
        loss.backward(retain_graph=True)
        self.label_updater.step()

    def optimize(self):
        self.pred_mse_updater.step()
    
    def optimize_with_label(self):
        self.label_updater.step()

    def __call__(self, input):
        move = 0
        press = 0

        X = self.observe(input) # 观察行动导致的环境变化，内部需要记忆

        if self.association:
            X = X + self.association
        if self.pred_ob:
            pred_loss = self.loss_fn_mse(self.pred_X, X) # 预测损失
            pred_loss.backward(retain_graph=True)
            ob_loss = self.loss_fn_mse(self.pred_ob, input)
            ob_loss.backward(retain_graph=True)

            self.pred_updater.step()
            self.pred_updater.zero_grad()

        self.param_collector.zero_grad()

        self.association = self.gen_assotiation(X) # 联想

        self.intention = self.update_intention(X, self.pred_X, self.last_action) # 由上一时刻的意图和当前时刻输入更新意图
        self.expecting = self.internal_to_observation(self.intention) # 网络的目标，目标明确时用于监督学习
        action = self.take_action(X, self.intention) # 由当前环境和意图采取行动

        self.pred_X = self.pred(X, action) # 预测候选行动导致的环境变化
        self.pred_ob = self.internal_to_observation(self.pred_X)
        with torch.no_grad():
            pred_mean = self.pred_X.mean()
        pred_mse = self.loss_fn_mse(self.pred_X, pred_mean) # 计算预测输入的均方差
        pred_mse.backward(retain_graph=True)

        # accept_candidate_action = self.decide(self.intention, X, self.pred_X) # 决定是否接受候选行动

        # if accept_candidate_action:
        #     self.action = self.action_candidate # 接受action
        # else:
        #     self.action = np.array([0,0]) # 观望
        self.last_action = action
        return action

if __name__ == '__main__':
    dataset_dir = "G:/DataSet/PyTorchDownload"
    dataset_name = 'MNIST'
    torch.autograd.set_detect_anomaly(True)
    dataset_reader = torchvision.datasets.__getattribute__(dataset_name)
    dataset_train = dataset_reader(dataset_dir, train=True,
        transform=transforms.ToTensor(), download=True
    )
    dataset_test = dataset_reader(dataset_dir, train=False,
        transform=transforms.ToTensor(), download=True
    )

    kernel_size = 7
    kernel_num = 256

    env_panel = Indicator()
    env_panel.reset()

    action = np.array([0, 0])
    count = 0
    while True:
        net = Agent(30, kernel_num, 5)

        observation, reward, _, _ = env_panel.step(action)
        act = net(torch.Tensor(observation))

        move = act[0:3]
        button = act[3:5]

        action = [torch.argmax(move), torch.argmax(button)]

        count += 1

        if reward <= 0:
            observation[1][env_panel.state_truth] = 1
            observation[2][env_panel.state_truth] = 1
            net.set_label(torch.Tensor(observation))
            net.optimize()
            net.optimize_with_label()
        else:
            net.optimize()
            print(f"count: {count}")
            count = 0



