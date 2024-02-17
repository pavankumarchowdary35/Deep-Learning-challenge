import torch.nn as nn
from torchvision import models
from data import ChristmasImages
import torchvision
import torch

class Network(nn.Module):

    def __init__(self):
        super().__init__()

        #############################
        # Initialize your network
        #############################
        self.weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b4(weights=self.weights)

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        in_features = 1792
        self.model.classifier = nn.Sequential(
        nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.1),
        #nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.1),
        nn.Linear(128, 8)
)

    def forward(self, x):

        #############################
        # Implement the forward pass
        #############################
        x = self.model(x)
        return x
    
   
    def save_model(self):

        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        torch.save(self.state_dict(), 'model_final.pkl')

    # def load_model(self, filename):
    #     # Load the model's weights
    #     self.model.load_state_dict(torch.load(filename, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
    
