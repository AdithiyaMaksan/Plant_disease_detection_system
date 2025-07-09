import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
import torch.nn as nn
from PIL import Image
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import sys

print("Enhanced Plant Disease Detection Model Evaluator")
print("-" * 50)

# Check if the model exists
model_path = "plant_disease_model_improved.pt"
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Please run the train_improved_enhanced.py script first to generate the model.")
    sys.exit(1)

# Check for GPU availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load class mapping
try:
    with open('class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
        idx_to_class = class_mapping['idx_to_class']
        # Convert string keys back to integers
        idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        class_to_idx = class_mapping['class_to_idx']
except FileNotFoundError:
    print("Warning: class_mapping.json not found. Will try to extract from model.")
    idx_to_class = None

# Load the model
try:
    checkpoint = torch.load(model_path, map_location=device)
    model_name = checkpoint.get('model_name', 'vgg16')
    num_classes = checkpoint.get('num_classes', len(idx_to_class) if idx_to_class else 0)
    
    # If class mapping wasn't loaded from file, try to get it from the model
    if idx_to_class is None:
        if 'idx_to_class' in checkpoint:
            idx_to_class = checkpoint['idx_to_class']
            class_to_idx = checkpoint['class_to_idx']
        else:
            print("Error: Could not find class mapping information.")
            sys.exit(1)
    
    # Create the model architecture
    if model_name == 'vgg16':
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
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully: {model_name} with {num_classes} classes")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Define transforms for evaluation
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Google image transform - more robust to handle various image sources
google_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to evaluate on test dataset
def evaluate_on_dataset(dataset_path="Dataset"):
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset directory {dataset_path} not found. Skipping dataset evaluation.")
        return
    
    try:
        # Load test dataset
        test_dataset = datasets.ImageFolder(dataset_path, transform=eval_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        print(f"\nEvaluating on {len(test_dataset)} images from {dataset_path}...")
        
        # Evaluation loop
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        # Generate confusion matrix
        class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
        cm = confusion_matrix(all_targets, all_preds)
        
        # Plot confusion matrix
        plt.figure(figsize=(15, 15))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        
        # Generate classification report
        report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
        
        # Save detailed metrics
        with open('evaluation_metrics.json', 'w') as f:
            json.dump(report, f, indent=4)
        
        print("\nDetailed metrics saved to evaluation_metrics.json")
        print("Confusion matrix saved to confusion_matrix.png")
        
        return accuracy, report
    
    except Exception as e:
        print(f"Error during dataset evaluation: {e}")
        return None, None

# Function to evaluate on test images directory
def evaluate_on_test_images(test_dir="../test_images"):
    if not os.path.exists(test_dir):
        print(f"Warning: Test images directory {test_dir} not found. Skipping test images evaluation.")
        return
    
    print(f"\nEvaluating on test images from {test_dir}...")
    
    results = []
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(test_dir, filename)
            try:
                # Load and preprocess the image
                image = Image.open(image_path).convert('RGB')
                image_tensor = google_transform(image).unsqueeze(0).to(device)
                
                # Make prediction
                with torch.no_grad():
                    output = model(image_tensor)
                    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                    
                    # Get top prediction
                    top_prob, top_idx = torch.max(probabilities, 0)
                    top_prob = top_prob.item() * 100
                    top_idx = top_idx.item()
                    predicted_class = idx_to_class[top_idx]
                    
                    # Get ground truth from filename (if available)
                    true_class = None
                    for class_name in class_to_idx.keys():
                        if class_name.lower().replace(' ', '_') in filename.lower():
                            true_class = class_name
                            break
                    
                    results.append({
                        'filename': filename,
                        'predicted_class': predicted_class,
                        'confidence': top_prob,
                        'true_class': true_class
                    })
                    
                    print(f"Image: {filename}")
                    print(f"  Predicted: {predicted_class} with {top_prob:.2f}% confidence")
                    if true_class:
                        print(f"  True class: {true_class}")
                        print(f"  {'✓ Correct' if predicted_class == true_class else '✗ Incorrect'}")
                    print()
            
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    # Calculate accuracy on test images with known ground truth
    test_images_with_truth = [r for r in results if r['true_class'] is not None]
    if test_images_with_truth:
        correct = sum(1 for r in test_images_with_truth if r['predicted_class'] == r['true_class'])
        accuracy = 100 * correct / len(test_images_with_truth)
        print(f"\nTest images accuracy: {accuracy:.2f}% ({correct}/{len(test_images_with_truth)})")
    
    # Save results to file
    with open('test_images_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Test images results saved to test_images_results.json")

# Function to evaluate on a single image (useful for Google images)
def evaluate_single_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found")
        return
    
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image_tensor = google_transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            print(f"\nPredictions for {os.path.basename(image_path)}:")
            print("-" * 50)
            for i in range(3):
                idx = top_indices[i].item()
                prob = top_probs[i].item() * 100
                class_name = idx_to_class[idx]
                print(f"{i+1}. {class_name} - {prob:.2f}% confidence")
    
    except Exception as e:
        print(f"Error processing image: {e}")

# Main evaluation
if __name__ == "__main__":
    # Check if a specific image is provided for evaluation
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        evaluate_single_image(image_path)
    else:
        # Evaluate on dataset if available
        evaluate_on_dataset()
        
        # Evaluate on test images
        evaluate_on_test_images()
        
        print("\nEvaluation complete!")
        print("To evaluate a specific image (e.g., from Google), run:")
        print("python evaluate_enhanced_model.py path/to/image.jpg")