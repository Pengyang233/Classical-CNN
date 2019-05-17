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
import DataUtils


if __name__ == '__main__':
    #Train or not
    is_train = False
    show = ToPILImage()
    #Set Parameters
    num_class = 10
    batch_size = 64
    lr = 0.02
    num_epochs = 5

    #Data preprocessing
    [loader_train, loader_test] = DataUtils.DataPreprocess1(batch_size=batch_size)

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
        while epoch_count < num_epochs:
            for data in loader_train:
                img, label = data

                #Convert label to one_hot format
                # label = label.view(-1,1)
                # label_onehot = torch.zeros(batch_size, num_class).scatter_(1,label,1)
                # label_onehot = label.long()
                # print(label_onehot)
                #Move to GPU
                img = img.cuda()
                label = label.cuda()
                # label_onehot = label_onehot.cuda()

                output = model(img)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_count+=1
            print('Epoch:{}, Loss:{:.4}'.format(epoch_count, loss))

        #Save the model
        torch.save(model.state_dict(),'./ModelPara/LeNet/lenet.pt')
        print('Model has been saved.')

    # Test Stage (Need to complete)
    if not is_train:
        #Load the model(The model must be defined beforehand if you use the function .load_state_dict())
        model.load_state_dict(torch.load('./ModelPara/LeNet/lenet.pt'))
        #-
        model.eval()
        sample_count = 0
        right_count = 0
        acc = 0
        test_loss = 0
        for data in loader_test:

            img, label = data
            img = img.cuda()
            label = label.cuda()

            output = model(img)
            loss = criterion(output, label)
            prediceted = torch.max(output.data,1)[1]

            sample_count += prediceted.size(0)
            '''Result:
               >> prediceted.size()  ->  torch.Size([64])
               >> prediceted.size(0) ->  64
               >> prediceted.size(1) ->  Raise Error
            '''
            right_count += (prediceted==label).sum().item()
            '''Result:
               >> (prediceted==label).sum()  ->  tensor(63, device='cuda:0')
               >> (prediceted==label).sum().item()  ->  63 
            '''
            acc = right_count/sample_count
        print('The accuracy is {}'.format(acc))
        print('The loss on test set is {}'.format(loss))







