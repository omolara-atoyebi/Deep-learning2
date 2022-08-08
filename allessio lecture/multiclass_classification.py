import torch as T
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from torch import optim

class MultiClass(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.input_layer = nn.Linear(784, 128)
        self.hidden1 = nn.Linear(128, 256)
        self.hidden2 = nn.Linear(256, 512)
        self.hidden3 = nn.Linear(512, 512)
        self.output = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        return self.output(x)


fake_mnist = T.rand(2, 784)
fake_labels = T.randint(0, 10, (10,))
model = MultiClass()

logits = model.forward(fake_mnist)

print(logits)

classes = F.softmax(logits, dim=1).argmax(dim=1)

print(classes)

