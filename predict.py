import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import PIL
import json
import numpy as np
import sys
import argparse


def get_args():
    # Initialization
    parser = argparse.ArgumentParser()
    # Add Arguments
    parser.add_argument('--image_path', type = str, default = 'flowers/test/1/image_06743.jpg', 
                        help = 'path to image file for prediction(default: flowers/test/1/image_06743.jpg)')
    parser.add_argument('--json_path', type = str, default = 'cat_to_name.json', 
                        help = 'path to label json file(default: cat_to_name.json)')
    parser.add_argument('--file_path', type = str, default = 'checkpoint.pth', 
                        help = 'path to checkpoint file(default: checkpoint.pth)')
    parser.add_argument('--topk', type = int, default = 5, 
                        help = 'Number of Topk(default: 5)')
    parser.add_argument('--gpu_mode', type = str, default = 'on', 
                        help = 'To use GPU, set it as "on". Else, CPU mode.')
    return parser.parse_args()


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


def load_cat_to_name(json_path):
    """Load Label Name from JSON file."""
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def load_checkpoint(file_path):
    # Load Checkpoint
    checkpoint = torch.load(file_path)
    # Model
    model = checkpoint['model'] 
    # Classifier of the model
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    # Optimizr
    lr = checkpoint['lr']
    optimizer = optim.Adam(model.classifier.parameters(), lr = lr)
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    
    return optimizer, model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with PIL.Image.open(image) as im:
        # Resizing perserving aspect ratios
        im = im.resize((224, 224), PIL.Image.ANTIALIAS)
        im = np.array(im) / 255
        # Normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        im = (im - mean) / std

        # TODO: Process a PIL image for use in a PyTorch model
        im = torch.from_numpy(im.transpose((2, 0, 1)))

        return im
    

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    image = process_image(image_path)
    image = image.to(device).unsqueeze(0).float()
    with torch.no_grad():
        probs, classes = torch.exp(model.forward(image)).topk(topk, dim=1)
    return probs, classes


def check_sanity(label_dict, title, image_path, model, device, topk=5):
    # Do prediction
    probs, classes = predict(image_path, model, device, topk)
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_int_list = [idx_to_class[i] for i in classes[0].tolist()]
    class_list = [label_dict[str(key)] for key in class_int_list]

    # Initialize Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Show image and label text.
    ax1 = imshow(process_image(image_path), ax=ax1, title = title)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.set_xticklabels('')
    ax1.set_yticklabels('')
    ax1.tick_params(axis='both', length=0)
    if title:
        ax1.set_title(title)
    ax1.axis('off')
    
    # Show Prediction Result
    ax2.barh(range(topk), probs[0].tolist())
    ax2.set_yticks(range(topk))
    ax2.set_yticklabels(class_list)
    
    ax2.set_xticks(np.arange(0, max(probs[0].tolist())*1.1, max(probs[0].tolist())*1.1/5))
    
    ax2.set_title(str(probs.grad_fn)[1:-1])
    
    ax2.invert_yaxis()
    plt.tight_layout()


def check_sanity_text(label_dict, title, image_path, model, topk):
    # Do prediction
    probs, classes = predict(image_path, model, topk)
    
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_int_list = [idx_to_class[i] for i in classes[0].tolist()]
    class_list = [label_dict[str(key)] for key in class_int_list]

    print(f'Label: {title}')
    for i in range(int(topk)):
        print(f'{probs[0].tolist()[i]}: {class_list[i]}')

# ------------------------------------------------------------------
# (3) Predict and show the result.
# ------------------------------------------------------------------

# check_sanity(label_dict = cat_to_name, 
#              title = cat_to_name['1'], 
#              image_path = 'flowers/test/1/image_06743.jpg', 
#              model = model, 
#              topk = 5)

# check_sanity(label_dict = cat_to_name, 
#              title = cat_to_name['1'], 
#              image_path = sys.argv[1], 
#              model = model, 
#              topk = 5)

# check_sanity_text(label_dict = cat_to_name, 
#                   title = cat_to_name['1'], 
#                   image_path = sys.argv[1], 
#                   model = model, 
#                   topk = 5)


def main():
    
    # Arg Parser
    args = get_args()
    file_path = args.file_path
    json_path = args.json_path
    image_path = args.image_path
    topk = args.topk
    gpu_mode = args.gpu_mode
    
    # Load Category Name
    cat_to_name = load_cat_to_name(json_path)
    
    # Load Checkpoint

    print(f'\nfile_path: {file_path}\n')
    print(f'\nRunning   >> Checkpoint Load Function')
    optimizer, model = load_checkpoint(file_path)
    print(f'\nFininshed ..')
    
    # START Prediction
    # ______________
    
    # Define device
    if gpu_mode == 'on':
        print(f"\nTrying GPU mode..\n")
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"\nGPU Available\n")
        else:
            device = torch.device('cpu')
            print(f"\nGPU Not Available\n")
    else:
        device = torch.device('cpu')
        
    model.to(device)
    print(device)
    
    # Predict Image
    print(f'\nRunning   >> Predict Function')
    probs, classes = predict(image_path, model, device, topk)
    print(f'\nFininshed ..')
    print(f'\nChecking  ..')
    print(f'\nPrinting  >> probs\n')
    print(probs)
    print(f'\nPrinting  >> classes\n')
    print(classes)
    
    print(f'\nPrinting  >> name of classes\n')
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    class_int_list = [idx_to_class[i] for i in classes[0].tolist()]
    class_list = [cat_to_name[str(key)] for key in class_int_list]
    print(class_list)
    print(f'\nPrinting End.\n')


if __name__ == "__main__":
    main()