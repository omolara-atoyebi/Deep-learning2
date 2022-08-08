from torchvision import datasets,transforms
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F


#download data
train_transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))])


trainset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True,train=True, transform=train_transform )
testset = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True,train=False, transform=train_transform )
trainloader = torch.utils.data.DataLoader(trainset,batch_size =64  , shuffle = True)
testloader = torch.utils.data.DataLoader(testset,batch_size = 64 , shuffle = True)


image, label = next(iter(trainloader))
images,labels = next(iter(testloader))


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim,output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim,400)
        self.fc2 = nn.Linear(400,200)
        self.fc3 = nn.Linear(200, 100)
        self.fc4 = nn.Linear(100,output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self,x):
        layer1 = self.fc1(x)
        act1 = self.relu(layer1)
        layer2 = self.fc2(act1)
        act2 = self.relu(layer2)
        layer3 = self.fc3(act2)
        act3 = self.relu(layer3)
        layer4 = self.fc4(act3)
        #output = self.softmax(layer4)
        return layer4

model = NeuralNetwork(input_dim=784 , output_dim = 10)



optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
epochs =10


#plot the data

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax

def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()

print_every = 40
train_losses = []
test_loses = []
accuracies =[]
for epoch in range(epochs):
    tr_losses = 0
    t_losses = 0
    print(f'{epoch+1}/{epochs}')
    for idx , (image,label) in enumerate (iter(trainloader)):
        img = image.reshape(image.shape[0],-1)
        optimizer.zero_grad()
        prediction = model.forward(img)
        train_loss = criterion(prediction , label)
        train_loss.backward()
        optimizer.step()
        tr_losses += train_loss.item()
        tra_loss = tr_losses/print_every

        train_losses.append(tra_loss)

    

        model.eval()
        with torch.no_grad():
            images = images.view(images.shape[0],-1)
            test_prediction = model.forward(images)
            test_loss = criterion(test_prediction,labels)
            t_losses += test_loss.item()
            tes_loss =t_losses/print_every
            test_loses.append(tes_loss)
            #top_p, top_class = test_prediction.topk(1,dim=1)
            classes =F.softmax(test_prediction, dim=1).argmax(dim=1)
    
            #tes_loss = t_losses/len(testloader)
            
            accuracy = sum(classes == labels)/len(test_prediction)
            accuracy = accuracy.item()
            #accuracy = accuracy/print_every
            accuracies.append(accuracy)
        # if idx % print_every == 0:
        #     print(f'epoch: {epoch} | loss: {tra_loss: .4f} | test loss: {test_loss} | accuracy: {accuracy.item()}')
        if idx % print_every == 0:
            print(f"\tIteration: {idx}\t Loss: {tra_loss:.4f}  test_loss:  {tes_loss:.4f}   accuracy: {accuracy *100}")
            running_loss = 0

        model.train()


plt.plot(train_losses, label='Train loss')
plt.plot(test_loses, label='Test loss')
plt.plot(accuracies, label='Accuracy')
plt.legend()
plt.show()