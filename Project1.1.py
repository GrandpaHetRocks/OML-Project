import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torch import nn

from torchvision import datasets, transforms
from torch import optim

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=True, transform=transform)
valset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)


print(trainset)
# Layer details for the neural network
input_size = 784
hidden_sizes = [6, 7]
output_size = 10

# Build a feed-forward network
model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
print(model)

criterion = nn.NLLLoss()


def run(epochs,ac_train,loss_train,ac_val,loss_val):
  for e in range(epochs):
      running_loss = 0
      model.train()
      for images, labels in trainloader:

          
          # Flatten MNIST images into a 784 long vector
          images = images.view(images.shape[0], -1)
          # images,labels=images.cuda(), labels.cuda()
      
          # Training pass
          optimizer.zero_grad()
          
          output = model(images)
          loss = criterion(output, labels)
          
          #This is where the model learns by backpropagating
          loss.backward()
          
          #And optimizes its weights here
          optimizer.step()
          
          running_loss += loss.item()
          
      else:
          print("Epoch {} - Training loss: {}".format(e, running_loss/len(valloader)))
          loss_train.append(running_loss/len(valloader))
  #print("\nTraining Time (in minutes) =",(time()-time0)/60)
          correct_count, all_count = 0, 0
          for images,labels in trainloader:
            # images,labels=images.cuda(), labels.cuda()
            for i in range(len(labels)):
              img = images[i].view(1, 784)
              # Turn off gradients to speed up this part
              with torch.no_grad():
                  model.eval()
                  logps = model(img)
          
              # Output of the network are log-probabilities, need to take exponential for probabilities
              ps = torch.exp(logps)
              probab = list(ps.numpy()[0])
              pred_label = probab.index(max(probab))
              true_label = labels.numpy()[i]
              if(true_label == pred_label):
                correct_count += 1
              all_count += 1
          
              
          print("Number Of Images Tested (Training)=", all_count)
          print("\nModel Accuracy (Training)=", (correct_count/all_count))
          ac_train.append((correct_count/all_count))

      correct_count, all_count = 0, 0
      running_loss = 0
      for images,labels in valloader:
        # images,labels=images.cuda(), labels.cuda()

        images = images.view(images.shape[0], -1)
        with torch.no_grad():
            model.eval()
            output = model(images)
            loss = criterion(output, labels)
            running_loss += loss.item()

        for i in range(len(labels)):
          img = images[i].view(1, 784)
          # Turn off gradients to speed up this part
          with torch.no_grad():
              model.eval()
              logps = model(img)
      
          # Output of the network are log-probabilities, need to take exponential for probabilities
          ps = torch.exp(logps)
          probab = list(ps.numpy()[0])
          pred_label = probab.index(max(probab))
          true_label = labels.numpy()[i]
          if(true_label == pred_label):
            correct_count += 1
          all_count += 1
      else:
          print("Epoch {} - Validation loss: {}".format(e, running_loss/len(valloader)))
          loss_val.append(running_loss/len(valloader))
          
          print("\nNumber Of Images Tested (Validation)=", all_count)
          print("Model Accuracy (Validation)=", (correct_count/all_count))
          ac_val.append((correct_count/all_count))
  return(ac_val,ac_train,loss_val,loss_train)

optimizer = optim.Adagrad(model.parameters(), lr=0.01, eps=1e-8)

epochs = 100

ac_train=[0]
loss_train=[0]
ac_val=[0]
loss_val=[0]
ac_val,ac_train,loss_val,loss_train=run(epochs,ac_train,loss_train,ac_val,loss_val)

print("Validation Accuracy")
print(ac_val)
print("Training Accuracy")
print(ac_train)
print("Validation Loss")
print(loss_val)
print("Training Loss")
print(loss_train)