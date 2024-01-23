import os
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import normalize
from PIL import Image
import torchvision.models as models
from torch import nn
import torchvision
from utils import transform_image
import torch.nn.functional as F
#from train import Network


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
    
class_labels = ['christmas_cookies', 'christmas_presents', 'christmas_tree', 'fireworks', 'penguin', 'reindeer', 'santa', 'snowman']

model1 = Network()
PATH = "models/densenet_B4_13.pth"
model1.load_state_dict(torch.load(PATH, map_location=device))
model1 = model1.to(device)
model1.eval()    


# Get a list of image file names in the test folder
test_folder = "data/data/val"
image_files = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]

print(len(image_files))
# Create a list to store predictions
prediction_label = []
prediction_class = []

with torch.inference_mode():
    # Loop through each image
    for image_file in image_files:
        image_path = os.path.join(test_folder, image_file)
        image = Image.open(image_path)
        #print(image.size)
        image = transform_image(image).to(device)
        prediction_logits = model1(image)
        pred_prob = torch.softmax(prediction_logits.squeeze(), dim=0)
        pred_prob = pred_prob.cpu()
        print("Prediction Logits:", pred_prob.shape)
        
        pred_label = pred_prob.argmax(dim=0).item()
        pred_class = class_labels[pred_label]

        #print(pred_label)

        # Append predictions to the list
        #print(prediction)
        prediction_label.append({"Image_ID": int(image_file.split('.')[0]), "Predicted_Label": pred_label})
        prediction_class.append({"Image_ID": int(image_file.split('.')[0]), "Prediction": pred_class})

        # print(len(prediction_label))
torch.set_grad_enabled(True)
# Create a DataFrame with predictions
df = pd.DataFrame(prediction_label)
df_1 = pd.DataFrame(prediction_class)

if len(df) < 160:
    print("Error: DataFrame should have at least 160 rows.")
else:
    df['Id'] = df['Image_ID']
    df['Category'] = df['Predicted_Label']
    df[['Id', 'Category']].to_csv('submission.csv', index=False, header=True)
    print(df.head())

df_1.to_csv('class_mapping.csv', index=False, header=True)

