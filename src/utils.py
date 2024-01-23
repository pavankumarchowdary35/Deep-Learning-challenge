import torch
from pathlib import Path
from torchvision import transforms
from PIL import Image
import io
import torchvision
import torch.nn as nn


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)
  
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

    # Save the entire model
    state_dict = model.state_dict()
    print(f"[INFO] Saving model state dictionary to: {model_save_path}")
    torch.save(obj=state_dict, f=model_save_path)



def transform_image(image):

    eval_transforms = transforms.Compose([
        transforms.Resize(384),
        #transforms.CenterCrop(456),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
        #transforms.RandomRotation(degrees=(-25, 25)),
        #transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), hue=(-0.2, 0.2)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if image.mode != 'RGB':
        image = image.convert('RGB')


    # Apply the rest of the transformations
    transformed_image = eval_transforms(image)

    # Add batch dimension
    transformed_image = transformed_image.unsqueeze(0)

    return transformed_image

