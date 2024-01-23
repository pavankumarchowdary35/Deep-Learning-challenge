from typing import List, Tuple
import os
import torch
import pandas as pd
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms.functional import normalize
from PIL import Image
import torchvision.models as models
from torch import nn



device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred(model: torch.nn.Module,
                        image_size: Tuple[int, int] = (232, 232),
                        #transform: torchvision.transforms = None,
                        device: torch.device=device,
                        test_path: str):
    

    image_files = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]
    
    predictions = []

# Turn on evaluation context manager
    with torch.inference_mode():
    # Loop through each image
      for image_file in image_files:
        # Load and preprocess the image
        image_path = os.path.join(test_folder, image_file)
        image = Image.open(image_path)
        if transform is not None:
          image_transform = transform
        else:
          image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


        image = image.unsqueeze(0).to(device)  # Add batch dimension and send to device

        # Forward pass to get predictions
        pred_logits = model(image)

        # Convert logits to probabilities
        pred_probs = torch.softmax(pred_logits, dim=1)

        # Get predicted labels
        pred_label = pred_probs.argmax(dim=1).item()

        # Append predictions to the list
        predictions.append({"Image_ID": int(image_file.split('.')[0]), "Predicted_Label": pred_label})

    return predictions    

    