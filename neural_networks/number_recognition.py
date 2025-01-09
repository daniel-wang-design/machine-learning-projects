import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

TRAINSET_PATH = 'number_recognition_trainset'
EVALSET_PATH = 'number_recognition_evalset'

trainset = datasets.MNIST(TRAINSET_PATH, download=True, train=True, transform=transform)
evalset = datasets.MNIST(EVALSET_PATH, download=True, train=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
evalloader = torch.utils.data.DataLoader(evalset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

print("Training the following model:")
print(model)
print("*******************************")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training model on {device}")
print("*******************************")

model = model.to(device=device)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

time0 = time()
EPOCHS = 50

for e in range(EPOCHS):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], -1).to(device)

        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(e, running_loss / len(trainloader)))
        
print("\nTraining Time (minutes) =",(time() - time0) / 60)

correct_count, all_count = 0, 0
for images,labels in evalloader:
  for i in range(len(labels)):
    img = images[i].view(1, 784).to(device)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    ps = ps.cpu()
    probab = list(ps.numpy(force=True)[0])
    pred_label = probab.index(max(probab))
    true_label = labels.numpy()[i]
    
    if(true_label == pred_label):
      correct_count += 1
      
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))

torch.save(model, './number_recognition_model.pt') 
