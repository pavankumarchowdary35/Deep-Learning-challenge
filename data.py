from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image
import os
import torch

class ChristmasImages(Dataset):
    def __init__(self, path, training=True, validation=False):
        super().__init__()
        self.training = training
        self.validation = validation
        self.path = path
        self.data_len = 0
        self.data_img_len = {}
        self.data_index_len = {}
        self.data_paths = {}

        if training and not validation:
            # Add training data transformations
            self.transform = transforms.Compose([
                # Add your training data transformations here
                transforms.Resize((512, 512)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(torch.nn.ModuleList([transforms.ColorJitter(), ]), p=0.3),
                transforms.RandomInvert(p=0.5),
                transforms.RandomGrayscale(p=0.3),
                transforms.ToTensor(),
                transforms.Normalize([0.4961, 0.4955, 0.5041], [0.3135, 0.3134, 0.3252])
            ])
            self._initialize_training_data(self.transform)

        elif not training and validation:
            # Use ImageFolder for validation data
            self.transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            validation_data = datasets.ImageFolder(self.path, transform=self.transform)
            self.data_len = len(validation_data)
        else:
            # Add testing data transformations
            self.transform = transforms.Compose([
                # Add your testing data transformations here
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            self._initialize_testing_data(self.transform)

    def _initialize_training_data(self, transform):
        # Use ImageFolder for training data
        train_data = datasets.ImageFolder(self.path, transform=transform)

        self.data_len = len(train_data)

    def _initialize_testing_data(self, transform):
        num_images = len(os.listdir(self.path))
        self.data_len = num_images
        dataPath = os.path.join(os.getcwd(), self.path)
        self.data_paths = [os.path.join(dataPath, imgName) for imgName in sorted(os.listdir(self.path), key=len)]
        self.transform = transform

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.training and not self.validation:
            # For training, directly use ImageFolder data
            return self._get_training_item(index)
        elif not self.training and self.validation:
            # For validation, directly use ImageFolder data
            return self._get_validation_item(index)
        else:
            # For testing, use custom logic
            return self._get_testing_item(index)

    def _get_training_item(self, index):

        return datasets.ImageFolder(self.path, transform=self.transform)[index]
        

    def _get_testing_item(self, index):
        img_path = self.data_paths[index]

        if not os.path.isfile(img_path):
            raise Exception("Image File not found: " + str(index) + ' Path: ' + img_path)

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        img_id = int(os.path.basename(img_path).split('.')[0])

        return image, img_id

    def _get_validation_item(self, index):
        # For validation, ImageFolder handles data loading
        return datasets.ImageFolder(self.path, transform=self.transform)[index]

    def getImagePath(self, index):
        img_class = ''
        for classname, imglen in self.data_index_len.items():
            if index < imglen:
                img_class = classname
                break

        if len(img_class) == 0:
            print('Index Class Error')

        img_class_path = self.data_paths[img_class]
        img_name = str(index - (self.data_index_len[img_class] - self.data_img_len[img_class])) + '.png'
        img_path = os.path.join(img_class_path, img_name)

        return img_path