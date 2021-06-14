import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
from datetime import datetime
import PIL
import json
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
import sys

# Basic usage: python train.py data_directory
# Basic usage: python train.py 'flowers'

# ------------------------------------------------------------------
# (1) Load Data
# ------------------------------------------------------------------


# data_dir = 'flowers'
data_dir = sys.argv[1]

dir_sets = {}

dir_sets['train'] = f'{data_dir}/train'
dir_sets['valid'] = f'{data_dir}/valid'
dir_sets['test'] = f'{data_dir}/test'

compose_sets = {}

compose_sets['train'] = [transforms.RandomRotation(30), 
                         transforms.RandomResizedCrop(224),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                              std = [0.229, 0.224, 0.225])]

compose_sets['valid'] = [transforms.Resize(255), 
                         transforms.CenterCrop(224), 
                         transforms.ToTensor(), 
                         transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                              std = [0.229, 0.224, 0.225])]

compose_sets['test'] = [transforms.Resize(255), 
                        transforms.CenterCrop(224), 
                        transforms.ToTensor(), 
                        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                             std = [0.229, 0.224, 0.225])]

image_datasets = {}

image_datasets['train'] = datasets.ImageFolder(dir_sets['train'], 
                                               transforms.Compose(compose_sets['train']))

image_datasets['valid'] = datasets.ImageFolder(dir_sets['valid'], 
                                               transforms.Compose(compose_sets['valid']))

image_datasets['test'] = datasets.ImageFolder(dir_sets['test'], 
                                              transforms.Compose(compose_sets['test']))

dataloaders = {}

dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], 
                                                   batch_size = 64, shuffle = True)

dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], 
                                                   batch_size = 64, shuffle = True)

dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], 
                                                  batch_size = 64, shuffle = True)


# ------------------------------------------------------------------
# (1) Label Mapping
# ------------------------------------------------------------------


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


# ------------------------------------------------------------------
# (2) Define a new, untrained feed-forward network
#     as a classifier, using ReLU activations and dropout.
# ------------------------------------------------------------------


class Classifier(nn.Module):
    def __init__(self, linear0, linear1, linear2, linear3, dropout_p):
        
        super().__init__()
      
        self.fc1 = nn.Linear(linear0, linear1)
        self.fc2 = nn.Linear(linear1, linear2)
        self.fc3 = nn.Linear(linear2, linear3)
      
        self.dropout = nn.Dropout(p = dropout_p)
  
    def forward(self, x):
      
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
       
        # Output
        x = F.log_softmax(self.fc3(x), dim = 1)
        return x

    
# ------------------------------------------------------------------
# (3) Setting for training
# ------------------------------------------------------------------


# Load Pretrained Model
model = models.vgg16(pretrained = True)


# Information for txt file
used_model = 'vgg16'
used_model_pretrained = 'True'


# Initial Input : 25088
#         Output: 102
linear0 = 25088
linear1 = 4096
linear2 = 1024
linear3 = 102
dropout_p = 0.2
model.classifier = Classifier(linear0, linear1, linear2, linear3, dropout_p)


# Setting for training
epochs = 5
lr = 0.003


# ------------------------------------------------------------------
# (4) Start Training
# ------------------------------------------------------------------


# Into CUDA mode
# Auto-Skip if the workspace doesn't support CUDA.
if torch.cuda.is_available():
    device = 'cuda'
    model.to(device)
else:
    device = 'cpu'
    model.to(device)

print(f"\n____________________",
      f"\n   {device}  MODE")
    

# Freeze features parameters not to backpropagate.
for param in model.features.parameters():
    param.requires_grad = False

# Negative log likelihood loss.
criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr)

# Initializing lists for each result
train_losses, valid_losses, test_losses = [], [], []
valid_accuracies, test_accuracies = [], []

# Start Time
file_name_suffix = datetime.now().strftime("_%m%d%H%M")
start_time = time.time()
started_time = datetime.now().strftime("%H:%M:%S")

print(f"\n____________________",
      f"\n   S  T  A  R  T",
      f"\n____________________",
      f"\nStart Time: {started_time}",
      f"\n____________________")

# Loop epochs
for e in range(epochs):

    # Initialization
    running_loss = 0
    
    for images, labels in dataloaders['train']:
        
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        log_ps = model.forward(images)
        loss = criterion(log_ps, labels)

        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    # Trun off grad
    with torch.no_grad():
        # into Evaluation Mode
        model.eval()

        # Initialization
        valid_loss = 0
        valid_accuracy = 0
        test_loss = 0
        test_accuracy = 0

        # Check Validation Loss and Accuraccy
        for images, labels in dataloaders['valid']:
            images, labels = images.to(device), labels.to(device)

            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            valid_loss += loss.item()

        # Check Test Loss and Accuraccy
        for images, labels in dataloaders['test']:
            images, labels = images.to(device), labels.to(device)

            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)

            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim = 1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            test_loss += loss.item()

    # Back into Train Mode
    model.train()

    # Save Result in each list
    train_losses.append(running_loss/len(dataloaders['train']))
    valid_losses.append(valid_loss/len(dataloaders['valid']))
    test_losses.append(test_loss/len(dataloaders['test']))
    valid_accuracies.append(valid_accuracy/len(dataloaders['valid']))
    test_accuracies.append(test_accuracy/len(dataloaders['test']))
    print(f"\nEpoch     : {e+1} / {epochs}\n",
          f"\nTrain Loss: {train_losses[-1]}",
          f"\nValid Loss: {valid_losses[-1]}",
          f"\nTest  Loss: {test_losses[-1]}",
          f"\nValid ACC.: {valid_accuracies[-1]}",
          f"\nTest  ACC.: {test_accuracies[-1]}")


# Ending
end_time = time.time()
seconds_elapsed = end_time - start_time
hours, rest = divmod(seconds_elapsed, 3600)
minutes, seconds = divmod(rest, 60)
avg_seconds_elapsed = seconds_elapsed / epochs
avg_hours, avg_rest = divmod(avg_seconds_elapsed, 3600)
avg_minutes, avg_seconds = divmod(avg_rest, 60)
ended_time = datetime.now().strftime("%H:%M:%S")

print(f"\n_______________________________\n",
      f"\nEnd   Time: {ended_time}",
      f"\nTotal Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
      f"\nAvg./Epoch: {int(avg_hours):02d}:{int(avg_minutes):02d}:{int(avg_seconds):02d}",
      f"\n_______________________________\n")


# Create a TXT file for the result
with open(f"temp_result{file_name_suffix}.txt", "a+") as f:
    f.write(f"\n_______________________________\n")
    f.write(f"\nTrain Mode: {device}")
    f.write(f"\nUsed Model: {used_model}")
    f.write(f"\nPretrained: {used_model_pretrained}")
    f.write(f"\nLinear0   : {linear0}")
    f.write(f"\nLinear1   : {linear1}")
    f.write(f"\nLinear2   : {linear2}")
    f.write(f"\nLinear3   : {linear3}")
    f.write(f"\nDropout_P : {dropout_p}")
    f.write(f"\nEpochs    : {epochs}")
    f.write(f"\nLearn Rate: {lr}")
    f.write(f"\n_______________________________\n")
    f.write(f"\nStart Time: {started_time}")
    f.write(f"\nEnd   Time: {ended_time}")
    f.write(f"\nTotal Time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
    f.write(f"\nAvg./Epoch: {int(avg_hours):02d}:{int(avg_minutes):02d}:{int(avg_seconds):02d}")
    f.write(f"\n_______________________________\n")
    for i in range(epochs):
        f.write(f"\nEpoch     : {i+1} / {epochs}")
        f.write(f"\nTrain Loss: {train_losses[i]}")
        f.write(f"\nValid Loss: {valid_losses[i]}")
        f.write(f"\nTest  Loss: {test_losses[i]}")
        f.write(f"\nValid ACC.: {valid_accuracies[i]}")
        f.write(f"\nTest  ACC.: {test_accuracies[i]}")
        f.write(f"\n_______________________________\n")
    f.write(f"\n_______________________________\n")
    f.write(f"\n* Class to IDX Map Added.\n")
    f.write(f"\n_______________________________\n")

        
# Add .class_to_idx
model.class_to_idx = image_datasets['train'].class_to_idx


# Save checkpoint
checkpoint = {'used_model': used_model, 
              'used_model_pretrained': used_model_pretrained,
              'linear0': linear0, 
              'linear1': linear1, 
              'linear2': linear2, 
              'linear3': linear3, 
              'classifier': model.classifier,
              'optimizer': optimizer,
              'optimizer_state': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

print(f"\n_______________________________\n")
torch.save(checkpoint, 'checkpoint.pth')
print(f"\ncheckpoint.pth file is saved!:D\n")


# Back to CPU mode
device = 'cpu'
model.to(device)


# Check training, validation, and test result.
plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.plot(test_losses, label='Test loss')
plt.legend(frameon=False)
print(f"\n_______________________FINISHED.\n")