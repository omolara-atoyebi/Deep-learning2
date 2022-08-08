import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary

# YOUR CODE HERE

class Neuronetwork(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1= nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, 30)
        self.fc3 = nn.Linear(30,20)
        self.fc3 = nn.Linear(20,1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        lay1 = self.relu(self.fc1(x))
        lay2 = self.relu(self.fc2(lay1))
        lay3 = self.relu(self.fc3(lay2))
        lay4 = self.sigmoid(lay3)
        return lay4


Model = Neuronetwork(input_dim = 3)

optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)
criterion = nn.MSELoss()

epochs = 100
        