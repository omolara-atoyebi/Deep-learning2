import torch as T
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

from torch import optim

class Binary(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.input_layer = nn.Linear(2, 8)
        self.hidden1 = nn.Linear(8, 16)
        self.hidden2 = nn.Linear(16, 32)
        self.hidden3 = nn.Linear(32, 32)
        self.output = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.input_layer(x))
        x = self.sigmoid(self.hidden1(x))
        x = self.sigmoid(self.hidden2(x))
        x = self.sigmoid(self.hidden3(x))
        return self.sigmoid(self.output(x))

data = pd.read_csv('data.csv', header=None)


x = T.from_numpy(data[[0, 1]].values).float()
y = T.from_numpy(data[[2]].values).float()

x_train = x[:80]
y_train = y[:80]

x_test = x[80:]
y_test = y[80:]

print(len(y_test))


model = Binary()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

epochs = 1000

train_losses = []
test_losses = []
accuracies = []

for epoch in range(epochs):

    # first step --> reset the gradients
    optimizer.zero_grad()

    # second step --> forward pass
    probs = model.forward(x_train)

    # third step --> computing loss
    loss = criterion(probs, y_train)

    # fourth step --> backward pass
    loss.backward()

    # fifth step --> updating the weights
    optimizer.step()

    train_losses.append(loss.item())

    

    # evaluating
    model.eval()
    with T.no_grad():
        test_probs = model.forward(x_test)
        # print(test_probs.shape, y_test.shape)
        test_loss = criterion(test_probs, y_test)
        test_losses.append(test_loss.item())
        classes = test_probs > 0.5

        accuracy = sum(classes == y_test)/len(test_probs)
        accuracies.append(accuracy.item())

    print(f'epoch: {epoch} | loss: {loss.item()} | accuracy: {accuracy.item()}')

plt.plot(train_losses, label='Train loss')
plt.plot(test_losses, label='Test loss')
plt.plot(accuracies, label='Accuracy')
plt.legend()
plt.show()

