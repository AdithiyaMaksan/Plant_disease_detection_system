import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.nn as nn
from torchvision import models

print("Plant Disease Detection - Google Images Predictor")
print("-" * 50)

# Check if the model exists
model_path = "plant_disease_model_improved.pt"
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Please run the train_improved_enhanced.py script first to generate the model.")
    sys.exit(1)

# Check if an image path was provided
if len(sys.argv) < 2:
    print("Usage: python predict_google_image.py <path_to_image>")
    print("Example: python predict_google_image.py ../test_images/apple_scab.JPG")
    
    # Check if test_images directory exists and suggest an example
    test_dir = "../test_images"
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG'))]
        if test_files:
            print(f"\nYou can try with a sample test image:")
            print(f"python predict_google_image.py {os.path.join(test_dir, test_files[0])}")
    
    sys.exit(1)

image_path = sys.argv[1]
if not os.path.exists(image_path):
    print(f"Error: Image file {image_path} not found")
    sys.exit(1)

# Load the model
print("Loading model...")
try:
    # Check for GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get class mapping
    if 'idx_to_class' in checkpoint:
        idx_to_class = checkpoint['idx_to_class']
        class_to_idx = checkpoint['class_to_idx']
    else:
        # Try to load from class_mapping.json
        try:
            with open('class_mapping.json', 'r') as f:
                class_mapping = json.load(f)
                idx_to_class = class_mapping['idx_to_class']
                # Convert string keys back to integers
                idx_to_class = {int(k): v for k, v in idx_to_class.items()}
                class_to_idx = class_mapping['class_to_idx']
        except FileNotFoundError:
            print("Error: Could not find class mapping information.")
            sys.exit(1)
    
    # Get model architecture and number of classes
    model_name = checkpoint.get('model_name', 'vgg16')
    num_classes = checkpoint.get('num_classes', len(idx_to_class))
    
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

# Define robust transform for Google Images
google_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and preprocess the image
try:
    print(f"\nProcessing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    
    # Apply transformations
    input_tensor = google_transform(image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_batch)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get top 3 predictions
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    
    # Display results
    print("\nTop 3 Predictions:")
    print("-" * 20)
    
    for i in range(3):
        idx = top3_indices[i].item()
        prob = top3_prob[i].item() * 100
        class_name = idx_to_class[idx].replace('_', ' ').title()
        print(f"{i+1}. {class_name}: {prob:.2f}% confidence")
    
    # Visualize the image with prediction
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(np.array(image))
    plt.title("Input Image")
    plt.axis('off')
    
    # Create a bar chart for the top 3 predictions
    plt.subplot(1, 2, 2)
    classes = [idx_to_class[idx.item()].replace('_', ' ').title() for idx in top3_indices]
    y_pos = np.arange(len(classes))
    
    plt.barh(y_pos, [prob.item() * 100 for prob in top3_prob], color='green')
    plt.yticks(y_pos, classes)
    plt.xlabel('Confidence (%)')
    plt.title('Top 3 Predictions')
    
    # Add confidence values as text
    for i, v in enumerate([prob.item() * 100 for prob in top3_prob]):
        plt.text(v + 1, i, f"{v:.1f}%", color='black', va='center')
    
    plt.tight_layout()
    
    # Save the visualization
    result_filename = f"prediction_result_{os.path.basename(image_path)}.png"
    plt.savefig(result_filename)
    print(f"\nVisualization saved as {result_filename}")
    
    # Show the plot if not in a headless environment
    try:
        plt.show()
    except:
        pass
    
    # Provide interpretation guidance based on confidence
    top_confidence = top3_prob[0].item() * 100
    if top_confidence > 80:
        print("\nInterpretation: High confidence prediction. The model is very certain about this diagnosis.")
    elif top_confidence > 50:
        print("\nInterpretation: Moderate confidence prediction. Consider the top 2 possibilities.")
    else:
        print("\nInterpretation: Low confidence prediction. The image may be unusual or contain features")
        print("not well represented in the training data. Consider trying another image or consulting an expert.")
        
    # Provide treatment suggestion placeholder
    # In a real application, this would connect to a database of treatments
    print(f"\nFor information on treating {classes[0]}, please consult our plant disease database.")
    
except Exception as e:
    print(f"Error processing image: {e}")
    sys.exit(1)