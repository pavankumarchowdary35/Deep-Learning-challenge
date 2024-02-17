from typing import Optional
import torch
from torch.utils.data import Dataset
import csv

from submission.data import ChristmasImages # This points to your submitted code
from tqdm.auto import tqdm

from typing import Optional
import torch
from torch.utils.data import DataLoader

from submission.model import Network # This points to your model


def evaluate():
    path = 'submission/val'
    model = Network()
    model.eval()
    model.load_state_dict(torch.load('submission/model_final.pkl', map_location=torch.device('cpu')))
    loader = DataLoader(TestSet(path), batch_size=1)
    
    accuracy = TestSet.evaluate(model, loader, device=torch.device('cuda:0'))
    
    # This should give you the accuracy of your model on the test set
    return accuracy

class TestSet(Dataset):
    
    def __init__(self, path):
        super().__init__()

        # Assure that your loader loads images in order 0.png, 1.png, 2.png, ...
        self.dataset = ChristmasImages(path , training=False)
        
        # You can assume that the csv file is in order of the images 0, 1, 2, ...
        with open('test_predictions_old.csv') as file:
            reader = csv.reader(file)
            next(reader)
            labels = {}
            for row in reader:
                labels[int(row[0])] = int(row[1])
        
        self.labels = labels
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Assure that the image idx when loaded is the same as the image idx in the csv file
        image = self.dataset[idx][0] 
        label = self.labels[idx]
        return image, label

    @staticmethod
    def evaluate(model, loader, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device('cuda:0')

        model = model.to(device)
        accuracy = 0.
        with torch.no_grad():
            it = tqdm(loader, desc='Evaluating', leave=False)
            for i, (image, label) in enumerate(it):
                image, label = image.to(device=device), label.to(device=device)
                _, prediction = model(image).max(dim=1)
                accuracy += (prediction == label).sum().item()

        model.cpu()
        accuracy /= len(loader)
        print(accuracy)
        return accuracy


if __name__ == "__main__":
    evaluate()