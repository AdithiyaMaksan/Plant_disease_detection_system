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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import torchvision.transforms.functional as TF
import PIL
from PIL import Image

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enhanced Data Augmentation
class AdvancedAugmentation:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            # Apply one of several augmentations
            aug_type = random.choice(['cutout', 'mixup', 'gaussian_noise'])
            
            if aug_type == 'cutout':
                # Implement cutout - randomly mask out a rectangle
                img = self._cutout(img)
            elif aug_type == 'gaussian_noise':
                # Add Gaussian noise
                img = self._add_gaussian_noise(img)
                
        return img
    
    def _cutout(self, img):
        # Convert to PIL if it's a tensor
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
            
        width, height = img.size
        cutout_size = min(width, height) // 4
        
        x = random.randint(0, width - cutout_size)
        y = random.randint(0, height - cutout_size)
        
        # Create a black rectangle
        img = np.array(img)
        img[y:y+cutout_size, x:x+cutout_size, :] = 0
        img = Image.fromarray(img)
        
        return img
    
    def _add_gaussian_noise(self, img):
        # Convert to PIL if it's a tensor
        if isinstance(img, torch.Tensor):
            img = TF.to_pil_image(img)
            
        img = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 15, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)
        
        return img

# Enhanced Data Augmentation for Training
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.05),  # Occasionally convert to grayscale
    transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    AdvancedAugmentation(p=0.3),  # Apply advanced augmentations
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Test/Validation Transform - Minimal processing for accurate evaluation
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Google Images Transform - More robust to handle various image sources
google_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset with augmentation
try:
    # Check if Dataset directory exists
    if not os.path.exists("Dataset"):
        print("\nWARNING: Dataset directory not found!")
        print("Please download the Plant Village dataset from:")
        print("https://data.mendeley.com/datasets/tywbtsjrjv/1")
        print("Extract it and ensure there's a 'Dataset' folder in the Model directory.")
        exit(1)
        
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
    
    # Create data loaders with increased workers and pin_memory for faster loading
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, sampler=train_sampler, num_workers=4, pin_memory=True
    )
    
    validation_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, sampler=validation_sampler, num_workers=4, pin_memory=True
    )
    
    # Get number of classes
    targets_size = len(train_dataset.class_to_idx)
    print(f"Training with {targets_size} classes")
    
    # Create class mapping
    idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
    class_to_idx = train_dataset.class_to_idx
    
    # Save class mapping for inference
    import json
    with open('class_mapping.json', 'w') as f:
        json.dump({'idx_to_class': idx_to_class, 'class_to_idx': class_to_idx}, f)
    
    print("Class mapping saved to class_mapping.json")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

# Enhanced Transfer Learning with VGG16
def create_transfer_model(num_classes, model_name='vgg16'):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        
        # Freeze early layers - only train later layers for better feature extraction
        for param in model.features[:24].parameters():  # Freeze more layers for better transfer learning
            param.requires_grad = False
        
        # Modify classifier with dropout for regularization
        n_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(n_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Higher dropout for better regularization
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    elif model_name == 'resnet50':
        # Alternative model architecture
        model = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze the last few layers
        for param in model.layer4.parameters():
            param.requires_grad = True
            
        # Replace the final fully connected layer
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
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
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(50176, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout
            nn.Linear(1024, K),
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(-1, 50176)  # Flatten
        out = self.dense_layers(out)
        return out

# Choose model type (transfer learning or custom CNN)
use_transfer_learning = True
model_name = 'vgg16'  # Options: 'vgg16', 'resnet50'

if use_transfer_learning:
    model = create_transfer_model(targets_size, model_name)
    print(f"Using {model_name} with transfer learning")
else:
    model = CNN(targets_size)
    print("Using custom CNN model")

model = model.to(device)

# Loss and optimizer with weight decay for regularization
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)  # AdamW with weight decay

# Learning rate scheduler - Cosine Annealing for better convergence
scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# Training function with early stopping and mixed precision training
def train_model(model, criterion, optimizer, scheduler, train_loader, validation_loader, epochs=30, patience=7):
    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    counter = 0
    best_model_state = None
    
    # Enable mixed precision training if available
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
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
            
            # Use mixed precision training if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
        scheduler.step()
        
        # Save metrics
        train_losses.append(avg_train_loss)
        validation_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        validation_accuracies.append(val_accuracy)
        
        # Early stopping check - now considers both loss and accuracy
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_val_loss = avg_val_loss
            counter = 0
            best_model_state = model.state_dict().copy()
            print(f"*** New best model saved with validation accuracy: {val_accuracy:.2f}% ***")
        else:
            counter += 1
        
        dt = datetime.now() - t0
        print(f"Epoch: {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Train Acc: {train_accuracy:.2f}% | "
              f"Val Acc: {val_accuracy:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
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
        model, criterion, optimizer, scheduler, train_loader, validation_loader, epochs=50, patience=10
    )
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': class_to_idx,
        'idx_to_class': idx_to_class,
        'model_name': model_name,
        'num_classes': targets_size
    }, "plant_disease_model_improved.pt")
    
    print("Model saved successfully!")
    
    # Plot training and validation loss
    plt.figure(figsize=(15, 6))
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
    
    # Create a function to predict from Google images
    def predict_from_image(image_path, model, transform=google_transform):
        model.eval()
        
        try:
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                prediction_idx = predicted.item()
                
                # Get class name
                class_name = idx_to_class[prediction_idx]
                
                # Get probabilities
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                confidence = probabilities[prediction_idx].item() * 100
                
                return class_name, confidence
        except Exception as e:
            return f"Error: {str(e)}", 0
    
    # Test the model on a sample image
    print("\nTesting model on a sample image...")
    test_images_dir = "../test_images"
    if os.path.exists(test_images_dir):
        test_files = os.listdir(test_images_dir)
        if test_files:
            test_image = os.path.join(test_images_dir, test_files[0])
            class_name, confidence = predict_from_image(test_image, model)
            print(f"Sample prediction: {class_name} with {confidence:.2f}% confidence")
    
    print("\nModel training complete! You can now use the model to predict plant diseases from any image source including Google images.")
    print("To use the improved model with the Flask app, run: python ../Flask\ Deployed\ App/use_improved_model.py")
    
except Exception as e:
    print(f"Error during training: {e}")

# Create a simple prediction script for Google images
with open('predict_google_image.py', 'w') as f:
    f.write('''
import torch
import torchvision.transforms as transforms
from PIL import Image
import json
import sys
import os

# Check if image path is provided
if len(sys.argv) < 2:
    print("Usage: python predict_google_image.py <path_to_image>")
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Error: Image file {image_path} not found")
    sys.exit(1)

# Load class mapping
try:
    with open('class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
        idx_to_class = class_mapping['idx_to_class']
        # Convert string keys back to integers
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
 except FileNotFoundError:
    print("Error: class_mapping.json not found. Please run training first.")
    sys.exit(1)

# Load the model
try:
    checkpoint = torch.load('plant_disease_model_improved.pt', map_location=torch.device('cpu'))
    model_name = checkpoint.get('model_name', 'vgg16')
    num_classes = checkpoint.get('num_classes', len(idx_to_class))
    
    # Create the model architecture
    if model_name == 'vgg16':
        from torchvision import models
        import torch.nn as nn
        
        model = models.vgg16(pretrained=False)
        n_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(n_features, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    elif model_name == 'resnet50':
        from torchvision import models
        import torch.nn as nn
        
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully: {model_name}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Define image transformation for Google images
google_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict function
def predict_disease(image_path):
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = google_transform(image).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            predictions = []
            for i in range(3):
                idx = top_indices[i].item()
                prob = top_probs[i].item() * 100
                class_name = idx_to_class[idx]
                predictions.append((class_name, prob))
            
            return predictions
    except Exception as e:
        return [(f"Error: {str(e)}", 0)]

# Make prediction
predictions = predict_disease(image_path)

# Print results
print(f"\nPredictions for {os.path.basename(image_path)}:")
print("-" * 50)
for i, (disease, confidence) in enumerate(predictions, 1):
    print(f"{i}. {disease} - {confidence:.2f}% confidence")
''')

print("\nCreated predict_google_image.py for easy prediction from Google images")
print("Usage: python predict_google_image.py <path_to_image>")