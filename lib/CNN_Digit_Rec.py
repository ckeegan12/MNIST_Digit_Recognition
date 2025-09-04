import pandas as pd
import numpy as np 
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets

train_dataset = torchvision.datasets.MNIST(
        root='./data',        
        train=True,           
        download=True,        
        transform=None  
)

mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)

mnist_testset.head()
