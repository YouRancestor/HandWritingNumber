import torch
from cv2 import cv2
import torchvision
from torch.utils import data
from torchvision import transforms

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


if __name__ == '__main__':
    dataset_dir = "G:/DataSet/PyTorchDownload"
    dataset_name = 'MNIST'

    dataset_reader = torchvision.datasets.__getattribute__(dataset_name)
    dataset_train = dataset_reader(dataset_dir, train=True,
        transform=transforms.ToTensor(), download=True
    )
    dataset_test = dataset_reader(dataset_dir, train=False,
        transform=transforms.ToTensor(), download=True
    )

    kernel_size = 7
    kernel_num = 256