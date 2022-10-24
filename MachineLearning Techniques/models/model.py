import torch
import torch.nn as nn
import torch.nn.functional as F

# architecture plan for titanic data
# (N,)
# n -> no of features

# (N,9) -> (N, 32) -> (N,16) -> (N,2)

class Classifier(nn.Module):

    def __init__(self,n):

        self.fc1=nn.Linear(n,32)
        self.fc2=nn.Linear(32,16)
        self.fc3=nn.Linear(16,2)
    
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
