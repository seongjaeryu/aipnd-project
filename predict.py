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
%matplotlib inline
%config InlineBackend.figure_format = 'retina'


# ------------------------------------------------------------------
# (1) Load Trained Model
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

# Load Checkpoint
checkpoint_path = 'checkpoint.pth'

# wrt. CUDA availability
if torch.cuda.is_available():
    checkpoint = torch.load(checkpoint_path)
else:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# Apply Model State Dict
model.load_state_dict(checkpoint['state_dict'])

# Apply Custom Options
model.class_to_idx = checkpoint['class_to_idx']

# Check Model
model.eval()


# ------------------------------------------------------------------
# (2) Define Functions
# ------------------------------------------------------------------


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


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = process_image(image_path)
    image = image.unsqueeze(0).float()
    probs, classes = torch.exp(model.forward(image)).topk(topk, dim=1)
    return probs, classes


def check_sanity(label_dict, title, image_path, model, topk=5):
    # Do prediction
    probs, classes = predict(image_path, model, topk)
    
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


# ------------------------------------------------------------------
# (3) Predict and show the result.
# ------------------------------------------------------------------


check_sanity(label_dict = cat_to_name, 
             title = cat_to_name['1'], 
             image_path = 'flowers/test/1/image_06743.jpg', 
             model = model, 
             topk = 5)