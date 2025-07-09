import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

print("Plant Disease Detection - Google Images Demo")
print("-" * 50)

# Check if the model exists
model_path = "plant_disease_model_improved.pt"
if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    print("Please run the train_improved_enhanced.py script first to generate the model.")
    sys.exit(1)

# Check if an image path was provided
if len(sys.argv) < 2:
    print("Usage: python google_image_demo.py <path_to_image>")
    print("Example: python google_image_demo.py ../test_images/apple_scab.JPG")
    
    # Check if test_images directory exists and suggest an example
    test_dir = "../test_images"
    if os.path.exists(test_dir):
        test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if test_files:
            print(f"\nYou can try with a sample test image:")
            print(f"python google_image_demo.py {os.path.join(test_dir, test_files[0])}")
    
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
    else:
        # Try to load from class_mapping.json
        import json
        try:
            with open('class_mapping.json', 'r') as f:
                class_mapping = json.load(f)
                idx_to_class = class_mapping['idx_to_class']
                # Convert string keys back to integers
                idx_to_class = {int(k): v for k, v in idx_to_class.items()}
        except FileNotFoundError:
            print("Error: Could not find class mapping information.")
            sys.exit(1)
    
    # Create the model architecture
    from torchvision import models
    import torch.nn as nn
    
    model_name = checkpoint.get('model_name', 'vgg16')
    num_classes = checkpoint.get('num_classes', len(idx_to_class))
    
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
    
    print(f"Model loaded successfully: {model_name}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Define transform for Google images
google_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and display the image
try:
    # Load image for display
    display_img = Image.open(image_path).convert('RGB')
    
    # Load and preprocess the image for prediction
    image = Image.open(image_path).convert('RGB')
    image_tensor = google_transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        # Prepare results for display
        predictions = []
        for i in range(3):
            idx = top_indices[i].item()
            prob = top_probs[i].item() * 100
            class_name = idx_to_class[idx]
            predictions.append((class_name, prob))
    
    # Display the image and predictions
    plt.figure(figsize=(10, 6))
    
    # Display image
    plt.subplot(1, 2, 1)
    plt.imshow(display_img)
    plt.title("Input Image")
    plt.axis('off')
    
    # Display predictions
    plt.subplot(1, 2, 2)
    y_pos = np.arange(len(predictions))
    bars = plt.barh(y_pos, [p[1] for p in predictions], align='center')
    plt.yticks(y_pos, [p[0] for p in predictions])
    plt.xlabel('Confidence (%)')
    plt.title('Top 3 Predictions')
    
    # Add percentage labels to bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f"{predictions[i][1]:.1f}%", va='center')
    
    plt.tight_layout()
    
    # Save the visualization
    result_image = 'prediction_result.png'
    plt.savefig(result_image)
    
    # Print results to console
    print(f"\nPredictions for {os.path.basename(image_path)}:")
    print("-" * 50)
    for i, (disease, confidence) in enumerate(predictions, 1):
        print(f"{i}. {disease} - {confidence:.2f}% confidence")
    
    print(f"\nVisualization saved as {result_image}")
    print("\nTo view the visualization, open the saved image file.")
    
    # Show plot if running in interactive environment
    plt.show()
    
except Exception as e:
    print(f"Error processing image: {e}")