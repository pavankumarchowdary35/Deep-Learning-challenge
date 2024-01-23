import os
import torch
import data_setup, engine, model_builder
import torchvision.models as models
from torch import nn
from pathlib import Path
import pandas as pd
import torchvision
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR

from torchvision import transforms
from utils import save_model

# Setup hyperparameters
NUM_EPOCHS = 75
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Setup directories
image_path = Path("data/data")
train_dir = "data/data/train"
val_dir = "data/data/val"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b4(weights=weights)
        #self.transform = weights.transforms()

        # for param in self.model.features.parameters():
        #    param.requires_grad = False

        # Modify the final fully connected layer in the classifier
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        in_features = 1792
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 512),nn.BatchNorm1d(512), nn.ReLU(),nn.Dropout(0.2) ,
            nn.Linear(512, 128),nn.BatchNorm1d(128),nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 8)) 
        

    def forward(self, x):
        x = self.model(x)
        return x
    

interpolation_mode = transforms.InterpolationMode.BICUBIC

additional_augmentations = transforms.Compose([
    transforms.Resize((384, 384), interpolation=interpolation_mode),
    #transforms.CenterCrop(528),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=(-20, 20)),
    #transforms.RandomAffine(degrees=0, shear=(-10, 10)),
    #transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


model = Network()
#automatic_transforms = model.transform

train_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    transform=additional_augmentations,
    batch_size=BATCH_SIZE
)

print(class_names)

torch.manual_seed(42)
torch.cuda.manual_seed(42)


# Set loss and optimizerSS
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

#scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.1)
#scheduler = CosineAnnealingLR(optimizer, T_max=75, eta_min=1e-8)
milestones = [50, 60, 70]
scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             scheduler=scheduler
             )

# Save the model with help from utils.py.
save_model(model=model,
                 target_dir="models",
                 model_name="densenet_B4_13.pth")


