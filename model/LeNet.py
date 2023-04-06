import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms, utils,datasets
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.optim as optim

class LeNet(nn.Module):
    
    def __init__(self, num_class=10) -> None:
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,6,kernel_size=5,stride=1,padding=2),
            nn.Tanh(),
            nn.AvgPool2d((2,2,)),
            nn.Conv2d(6,16,kernel_size=5,padding=2),
            nn.Tanh(),
            nn.AvgPool2d((2,2)),
            nn.Conv2d(16,120,kernel_size=5,padding=2),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(7680,84),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(84,num_class)
        )
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x=self.features(x)
        x= torch.flatten(x,1)
        x= self.classifier(x)
        return x
