import os
import shutil
import sys

print("Plant Disease Detection - Flask App Updater")
print("-" * 50)

# Check if the improved model exists
model_path = "plant_disease_model_improved.pt"
flask_app_dir = "../Flask Deployed App"
flask_model_path = os.path.join(flask_app_dir, model_path)

if not os.path.exists(model_path):
    print(f"Error: Improved model not found at {model_path}")
    print("Please run the train_improved_enhanced.py script first to generate the improved model.")
    sys.exit(1)

# Copy the improved model to the Flask app directory
print(f"Copying improved model to Flask app directory...")
try:
    shutil.copy2(model_path, flask_model_path)
    print(f"Successfully copied model to {flask_model_path}")
except Exception as e:
    print(f"Error copying model: {e}")
    sys.exit(1)

# Copy the class mapping file
class_mapping_path = "class_mapping.json"
flask_class_mapping_path = os.path.join(flask_app_dir, class_mapping_path)

if os.path.exists(class_mapping_path):
    try:
        shutil.copy2(class_mapping_path, flask_class_mapping_path)
        print(f"Successfully copied class mapping to {flask_class_mapping_path}")
    except Exception as e:
        print(f"Error copying class mapping: {e}")

# Create backup of original app.py
app_py_path = os.path.join(flask_app_dir, "app.py")
if os.path.exists(app_py_path):
    print("Creating backup of original app.py...")
    shutil.copy2(app_py_path, app_py_path + ".backup")
    print("Backup created as app.py.backup")

# Update app.py to use the improved model and support Google images
with open(app_py_path, "r") as f:
    app_code = f.read()

# Replace the model loading line
if "plant_disease_model_1_latest.pt" in app_code:
    updated_code = app_code.replace(
        "plant_disease_model_1_latest.pt", 
        "plant_disease_model_improved.pt"
    )
    
    # Add support for Google images by modifying the transformation
    if "transforms.Resize((224, 224))" in updated_code:
        updated_code = updated_code.replace(
            "transforms.Compose([transforms.Resize((224, 224)),",
            "transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),"
        )
    
    # Write the updated code back to app.py
    with open(app_py_path, "w") as f:
        f.write(updated_code)
    
    print("Successfully updated app.py to use the improved model!")
    print("Added support for Google images with enhanced preprocessing")

# Create a README file with instructions
readme_content = """
# Enhanced Plant Disease Detection Model

## Improvements Implemented

1. **Advanced Data Augmentation**
   - Random resized cropping with variable scales
   - Random horizontal and vertical flips
   - Random rotation (up to 30 degrees)
   - Color jittering with enhanced parameters
   - Random grayscale conversion
   - Random perspective and affine transformations
   - Advanced augmentations (cutout, Gaussian noise)

2. **Enhanced Transfer Learning**
   - VGG16 pre-trained on ImageNet with optimized layer freezing
   - Deeper classifier with multiple dropout layers
   - Support for alternative architectures (ResNet50)

3. **Training Optimizations**
   - AdamW optimizer with weight decay for better regularization
   - Cosine annealing learning rate scheduler
   - Mixed precision training for faster convergence
   - Early stopping based on validation accuracy
   - Improved model checkpointing

4. **Google Images Support**
   - Specialized preprocessing for external images
   - Robust prediction pipeline for various image sources
   - Top-3 predictions with confidence scores

## How to Use

### Training the Enhanced Model

```bash
python train_improved_enhanced.py
```

The script will:
1. Load the dataset with advanced augmentation
2. Create an enhanced transfer learning model
3. Train with optimized parameters and early stopping
4. Save the improved model and class mapping
5. Generate training curves

### Predicting from Google Images

```bash
python predict_google_image.py path/to/image.jpg
```

### Updating the Flask App

```bash
python update_flask_app.py
```

This will update the Flask app to use the improved model and support Google images.

## Troubleshooting

1. **Dataset not found**
   - Ensure the Dataset directory is in the Model folder
   - Download from: https://data.mendeley.com/datasets/tywbtsjrjv/1

2. **Model training errors**
   - Check GPU memory availability
   - Reduce batch size if needed
   - Ensure dataset is properly structured

3. **Prediction issues with Google images**
   - Ensure image is in a standard format (JPG, PNG)
   - Check image quality and lighting
   - Try preprocessing the image manually if needed
"""

with open("ENHANCED_MODEL_README.md", "w") as f:
    f.write(readme_content)

print("\nCreated ENHANCED_MODEL_README.md with detailed instructions")
print("\nTo use the improved model with the Flask app:")
print("1. First train the model by running: python train_improved_enhanced.py")
print("2. Then update the Flask app by running: python update_flask_app.py")
print("3. Start the Flask app: python ../Flask\ Deployed\ App/app.py")
print("\nTo predict from Google images directly:")
print("python predict_google_image.py path/to/image.jpg")