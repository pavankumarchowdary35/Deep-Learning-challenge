import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt
from data import ChristmasImages
from model import Network
from torch.utils.data import default_collate
from torchvision.transforms import v2
from torch.optim.lr_scheduler import ExponentialLR

cutmix = v2.CutMix(num_classes=8)
mixup = v2.MixUp(num_classes=8)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

def collate_fn(batch):
    return cutmix_or_mixup(*default_collate(batch))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
batch_size = 16
learning_rate = 0.001
num_epochs = 100

# Create DataLoader
train_dataset = ChristmasImages(path="data/data/train", training=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn)

validation_dataset = ChristmasImages(path="val", training=False, validation=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Initialize model
model = Network().to(device)

# Use DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = nn.DataParallel(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler (ReduceLROnPlateau for early stopping)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
scheduler = ExponentialLR(optimizer, gamma=0.9)

train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []

best_train_loss = float('inf')  # Initialize with positive infinity
best_validation_accuracy = 0.0  # Track the best validation accuracy
early_stop_patience = 7
early_stop_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for inputs, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if len(labels.size()) > 1:
            labels = torch.argmax(labels, dim=1)

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

    # Save the model based on the loss at the end of the epoch
    average_loss = total_loss / len(train_dataloader)
    accuracy = correct_predictions / total_samples

    train_losses.append(average_loss)
    train_accuracies.append(accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Validation
    model.eval()
    validation_loss = 0.0
    validation_correct_predictions = 0
    validation_total_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(validation_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs} (Validation)"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            validation_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            validation_correct_predictions += (predicted == labels).sum().item()
            validation_total_samples += labels.size(0)

    # Calculate validation accuracy and loss
    validation_average_loss = validation_loss / len(validation_dataloader)
    validation_accuracy = validation_correct_predictions / validation_total_samples

    validation_losses.append(validation_average_loss)
    validation_accuracies.append(validation_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs} (Validation), Loss: {validation_average_loss:.4f}, Accuracy: {validation_accuracy:.4f}")

    # Update learning rate scheduler with training loss
    scheduler.step()

    # Early stopping check based on training loss
    if average_loss < best_train_loss:
        best_train_loss = average_loss
        early_stop_counter = 0  # Reset the counter if the training loss improves
    else:
        early_stop_counter += 1

    # Save the model with the best validation accuracy
    if validation_accuracy > best_validation_accuracy:
        best_validation_accuracy = validation_accuracy
        model.module.save_model()

    # Early stopping check
    if early_stop_counter >= early_stop_patience:
        print(f"No improvement in training loss for {early_stop_patience} epochs. Early stopping.")
        break

# Save the final trained model
# model.module.save_model()
print("Training complete.")

# Plotting
epochs_range = range(1, len(train_losses) + 1)

# Plot Training Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Train Loss', marker='o')
plt.plot(epochs_range, validation_losses, label='Validation Loss', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plot Training Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(epochs_range, validation_accuracies, label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Save the plots
plt.savefig('training_plots.png')
plt.show()
