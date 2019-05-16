'''
Author:Pengyang
Date:
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToPILImage

import matplotlib.pylab as plt

#Choose medel file
import LeNet


if __name__ == '__main__':
    #Train or not
    is_train = True
    show = ToPILImage()
    #Set Parameters
    batch_size = 64
    lr = 0.02
    num_epochs = 20

    #Data Augumentation
    mytansform = transforms.Compose([
        transforms.RandomRotation(45),
        transforms.Resize([32,32]),
        transforms.ToTensor()        
    ])

    #Generate Dataset
    mnist_train = datasets.MNIST(root='./dataset/mnist', train=True, download=True,transform=mytansform)
    mnist_test = datasets.MNIST(root='./dataset/mnist', train=False, download=True,transform=mytansform)
    loader_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    loader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

    #Chose Model
    model = LeNet.LeNet()
    model = model.cuda(device=0)

    #Chose Criterion and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=lr)

    #Train Stage
    if is_train:
        epoch_count = 0
        model.train(mode=True)
        for data in loader_train:
            img, label = data
            img = img.cuda()
            label = label.cuda()

            output = model(img)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_count+=1

            if epoch_count%50 == 0:
                print('Epoch:{}, Loss:{:.4}'.format(epoch_count, loss))

    # Test Stage
    if not is_train:
        model.eval()
        sample_count = 0
        right_count = 0
        acc = 0
        test_loss = 0
        for data in loader_test:
            sample_count +=1

            img, label = data

            output = model(img)
            loss = criterion(output, label)
            if output == label:
                right_count += 1

            acc = right_count/sample_count







