#import the needed libraries

import data_handler as dh           # yes, you can import your code. Cool!
import torch
import torch.optim as optim
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from model import Model , criterion,optimizer, epochs



# Remember to validate your model: model.eval .........with torch.no_grad() ......model.train
x_train, x_test, y_train, y_test = dh.load_data()


for epoch in range(epochs):
    optimizer.zero_grad()
    prediction = Model.forward(x_train)
    Train_Loss = criterion(prediction, y_train)
    Train_Loss.backward()
    optimizer.step()


    Model.eval()
    with torch.no_grad():
        prediction_test = Model.forward(x_test)
        test_loss = criterion(prediction_test, y_test)
    Model.train()
