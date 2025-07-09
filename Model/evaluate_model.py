import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import sys

# Check if model path is provided
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = "plant_disease_model_improved.pt"  # Default to improved model

print(f"Evaluating model: {model_path}")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the CNN model
sys.path.append(os.path.join(os.path.dirname(__file__), '../Flask Deployed App'))
import CNN

# Data transformation for evaluation
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
try:
    test_dataset = datasets.ImageFolder("Dataset", transform=test_transform)
    print(f"Dataset loaded with {len(test_dataset)} images")
    
    # Create indices for test set (20% of data)
    dataset_size = len(test_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.2 * dataset_size))
    
    # Use fixed seed for reproducibility
    np.random.seed(42)
    np.random.shuffle(indices)
    
    test_indices = indices[:split]
    test_sampler = SubsetRandomSampler(test_indices)
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=32, sampler=test_sampler, num_workers=4
    )
    
    # Get class names
    class_names = test_dataset.classes
    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# Load model
try:
    # Create model with the same number of classes as in the dataset
    model = CNN.CNN(len(class_names))
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Evaluate model
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    class_correct = {i: 0 for i in range(len(class_names))}
    class_total = {i: 0 for i in range(len(class_names))}
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # Overall accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i]
                pred = predicted[i]
                if label == pred:
                    class_correct[label.item()] += 1
                class_total[label.item()] += 1
            
            # Save predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for i in range(len(class_names)):
        if class_total[i] > 0:
            class_accuracy[i] = 100 * class_correct[i] / class_total[i]
            print(f"Accuracy of {idx_to_class[i]}: {class_accuracy[i]:.2f}%")
    
    return accuracy, all_preds, all_labels

# Run evaluation
accuracy, all_preds, all_labels = evaluate_model(model, test_loader, device)

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(20, 16))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[idx_to_class[i] for i in range(len(class_names))],
            yticklabels=[idx_to_class[i] for i in range(len(class_names))])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix - Accuracy: {accuracy:.2f}%')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()

# Save the confusion matrix
output_file = f"confusion_matrix_{os.path.basename(model_path).split('.')[0]}.png"
plt.savefig(output_file)
print(f"Confusion matrix saved to {output_file}")

# Generate classification report
report = classification_report(all_labels, all_preds, 
                             target_names=[idx_to_class[i] for i in range(len(class_names))],
                             output_dict=True)

# Convert to DataFrame and save
report_df = pd.DataFrame(report).transpose()
report_file = f"classification_report_{os.path.basename(model_path).split('.')[0]}.csv"
report_df.to_csv(report_file)
print(f"Classification report saved to {report_file}")

print("\nTo compare models, run:")
print("python evaluate_model.py plant_disease_model_1_latest.pt")
print("python evaluate_model.py plant_disease_model_improved.pt")