import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data Augmentation
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset with augmentation
try:
    train_dataset = datasets.ImageFolder("Dataset", transform=train_transform)
    test_dataset = datasets.ImageFolder("Dataset", transform=test_transform)
    print(f"Dataset loaded successfully with {len(train_dataset)} images")
    print(f"Number of classes: {len(train_dataset.classes)}")
    
    # Create data indices for training and validation splits
    dataset_size = len(train_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    
    # Shuffle indices
    np.random.shuffle(indices)
    
    train_indices, validation_indices = indices[split:], indices[:split]
    
    # Create samplers
    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(validation_indices)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, sampler=train_sampler, num_workers=4
    )
    
    validation_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, sampler=validation_sampler, num_workers=4
    )
    
    # Get number of classes
    targets_size = len(train_dataset.class_to_idx)
    print(f"Training with {targets_size} classes")
    
    # Create class mapping
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Transfer Learning with VGG16
def create_transfer_model(num_classes):
    model = models.vgg16(pretrained=True)
    
    # Freeze early layers
    for param in model.features[:20].parameters():
        param.requires_grad = False
    
    # Modify classifier
    n_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(n_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    
    return model

# Original CNN model
class CNN(nn.Module):
    def __init__(self, K):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # conv2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # conv3
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            # conv4
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
        )

        self.dense_layers = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176)  # Flatten
        out = self.dense_layers(out)
        return out

# Choose model type (transfer learning or custom CNN)
use_transfer_learning = True

if use_transfer_learning:
    model = create_transfer_model(targets_size)
    print("Using VGG16 with transfer learning")
else:
    model = CNN(targets_size)
    print("Using custom CNN model")

model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# Training function with early stopping
def train_model(model, criterion, optimizer, scheduler, train_loader, validation_loader, epochs=25, patience=7):
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    
    best_val_loss = float('inf')
    counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        t0 = datetime.now()
        model.train()
        train_loss = []
        train_correct = 0
        train_total = 0
        
        # Training loop
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # Validation loop
        model.eval()
        validation_loss = []
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in validation_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                validation_loss.append(loss.item())
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate average losses and accuracies
        avg_train_loss = np.mean(train_loss)
        avg_val_loss = np.mean(validation_loss)
        train_accuracy = 100 * train_correct / train_total
        val_accuracy = 100 * val_correct / val_total
        
        # Update learning rate scheduler
        scheduler.step(avg_val_loss)
        
        # Save metrics
        train_losses.append(avg_train_loss)
        validation_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
        else:
            counter += 1
        
        dt = datetime.now() - t0
        print(f"Epoch: {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | "
              f"Val Acc: {val_accuracy:.2f}% | "
              f"Duration: {dt}")
        
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, validation_losses, train_accuracies, validation_accuracies

# Train the model
print("Starting training...")
try:
    model, train_losses, validation_losses, train_accuracies, validation_accuracies = train_model(
        model, criterion, optimizer, scheduler, train_loader, validation_loader, epochs=30
    )
    
    # Save the model
    torch.save(model.state_dict(), "plant_disease_model_improved.pt")
    print("Model saved successfully!")
    
    # Plot training and validation loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(validation_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
except Exception as e:
    print(f"Error during training: {e}")