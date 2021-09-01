import torch
from cv2 import cv2
import torchvision
from torch.utils import data
from torchvision import transforms



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