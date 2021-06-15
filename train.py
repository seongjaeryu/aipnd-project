import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from datetime import datetime
import json
import sys
import argparse


def get_args():
    # Initialization
    parser = argparse.ArgumentParser()
    # Add Arguments
    parser.add_argument('--data_dir', type = str, default = 'flowers', 
                        help = 'path to data folder(default: flowers)')
    parser.add_argument('--json_path', type = str, default = 'cat_to_name.json', 
                        help = 'path to label json file(default: cat_to_name.json)')    
    parser.add_argument('--gpu_mode', type = str, default = 'on', 
                        help = 'To use GPU, set it as "on". Else, CPU mode.')
    parser.add_argument('--model_name', type = str, default = 'vgg16', 
                        help = 'name of pretrained model to use(options: vgg16(default), alexnet, resnet18)')
    parser.add_argument('--hidden', type = int, nargs = 2, default = [4096, 1024], 
                        help = 'list of 2 natural numbers for each of 2 hidden layers (defalut: [4096, 1024])')
    parser.add_argument('--output', type = int, default = 102, 
                        help = 'number of classes(default: 102)')    
    parser.add_argument('--dropout_p', type = float, default = 0.2, 
                        help = '0 <= float for dropout(default: 0.2) < 1')
    parser.add_argument('--lr', type = float, default = 0.003, 
                        help = '0 <= float for learning rate(default: 0.003)')
    parser.add_argument('--epochs', type = int, default = 5, 
                        help = 'natural number for number of epoch(default: 5)')
    return parser.parse_args()

    
def load_data(data_dir):
    """Load data and create sets and loaders for data.
    
    Argument(1):
        data_dir -- (str) directory path of data folders.
    
    Return(4):    
        dir_sets -- (dic)
            Value: (str) directory path
            Keys : 'train', 'valid', 'test'
        compose_sets -- (dic)
            Value: (list) of transforms method(s)
            Keys : 'train', 'valid', 'test'
        image_datasets -- (dic)
            Value: ImageFolder method
            Keys : 'train', 'valid', 'test'
        dataloaders -- (dic)
            Value: DataLoader method
            Keys : 'train', 'valid', 'test'
    """
    # Define dir_set
    dir_sets = {}
    dir_sets['train'] = f'{data_dir}/train'
    dir_sets['valid'] = f'{data_dir}/valid'
    dir_sets['test'] = f'{data_dir}/test'
    # Define compose_set
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
    # Define image_datasets
    image_datasets = {}
    image_datasets['train'] = datasets.ImageFolder(dir_sets['train'], 
                                                   transforms.Compose(compose_sets['train']))
    image_datasets['valid'] = datasets.ImageFolder(dir_sets['valid'], 
                                                   transforms.Compose(compose_sets['valid']))
    image_datasets['test'] = datasets.ImageFolder(dir_sets['test'], 
                                                  transforms.Compose(compose_sets['test']))
    # Define dataloaders
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], 
                                                       batch_size = 64, shuffle = True)
    dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], 
                                                       batch_size = 64, shuffle = True)
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], 
                                                      batch_size = 64, shuffle = True)
    
    return dir_sets, compose_sets, image_datasets, dataloaders


def load_cat_to_name(json_path):
    """Load Label Name from JSON file."""
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    
    return cat_to_name


class Classifier(nn.Module):
    """Classifier with 3 fully connected layers"""
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


def load_model(model_name):
    if model_name == 'vgg16':
        linear0 = 25088
        model = models.vgg16(pretrained = True)
        return model, linear0
    elif model_name == 'alexnet':
        linear0 = 9216
        model = models.alexnet(pretrained = True)
        return model, linear0
    elif model_name == 'resnet18':
        linear0 = 2208
        model = models.resnet18(pretrained = True)
        return model, linear0
    else:
        print(f"\n'{model_name}' is not supported.")
        print(f"\nSupport Models: 'vgg16', 'alexnet', 'resnet18'")
        sys.exit()


def main():    
    # Arg Parser
    args = get_args()
    data_dir = args.data_dir
    json_path = args.json_path
    gpu_mode = args.gpu_mode
    model_name = args.model_name
    linear1, linear2 = args.hidden
    linear3 = args.output
    dropout_p = args.dropout_p
    lr = args.lr
    epochs = args.epochs
    
    # Load Data
    dir_sets, compose_sets, image_datasets, dataloaders = load_data(data_dir)
    
    # Load Label Name
    cat_to_name = load_cat_to_name(json_path)
    
    # Load Model
    model, linear0= load_model(model_name)
    
    # Update Classifier
    model.classifier = Classifier(linear0, linear1, linear2, linear3, dropout_p)
    
    # START TRAINING
    # ______________
    
    # Into GPU mode
    if gpu_mode == 'on':
        print(f"\nTrying GPU mode..\n")
        if torch.cuda.is_available():
            device = 'cuda'
            model.to(device)
            print(f"\nGPU training mode changed successfully..\n")
        else:
            device = 'cpu'
            model.to(device)
            print(f"\nGPU training mode failed..\n")
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
        f.write(f"\nUsed Model: {model_name}")
        f.write(f"\nLinear0   : {linear0} (Preset for each Model)")
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
    checkpoint = {'model_name': model_name,
                  'model': model,
                  'linear0': linear0, 
                  'linear1': linear1, 
                  'linear2': linear2, 
                  'linear3': linear3, 
                  'classifier': model.classifier,
                  'optimizer_state': optimizer.state_dict(),
                  'lr': lr,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    print(f"\n_______________________________\n")
    torch.save(checkpoint, 'checkpoint.pth')
    print(f"\ncheckpoint.pth file is saved!:D\n")

    # End Message
    print(f"\n_______________________FINISHED.\n")


if __name__ == "__main__":
    main()