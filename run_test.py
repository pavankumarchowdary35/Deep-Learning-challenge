
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data import ChristmasImages
from model import Network
import pandas as pd

class_labels = ['christmas_cookies', 'christmas_presents', 'christmas_tree', 'fireworks', 'penguin', 'reindeer', 'santa', 'snowman']

def test(model, test_dataloader, device, class_labels, output_csv='predictions.csv'):
    model.eval()

    prediction_label = []
    prediction_class = []

    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for img_id, pred_label in zip(labels, predicted):
                prediction_label.append({"Id": int(img_id.item()), "Category": pred_label.item()})
                prediction_class.append({"Image_ID": int(img_id.item()), "Prediction": class_labels[pred_label.item()]})

    # Create DataFrames
    df = pd.DataFrame(prediction_label)
    df_class = pd.DataFrame(prediction_class)

    # Save to CSV files
    df.to_csv(output_csv, index=False, header=True)
    df_class.to_csv('class_mapping.csv', index=False, header=True)

    print(f"Predictions saved to {output_csv}")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
batch_size = 32

# Transformations

# Load the trained model
test_model = Network().to(device)
#test_model.load_model('model_final.pkl')
test_model.load_state_dict(torch.load('model_final.pkl', map_location=torch.device('cpu')))

# Create DataLoader for testing
test_dataset = ChristmasImages(path="data/data/val", training=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Test the model
test(test_model, test_dataloader, device, class_labels, output_csv='test_predictions_new.csv')

