import torch
import os
import shutil
import sys

# Check if the improved model exists
model_path = "../Model/plant_disease_model_improved.pt"
app_model_path = "plant_disease_model_improved.pt"

print("Plant Disease Detection Model Updater")
print("-" * 40)

if not os.path.exists(model_path):
    print(f"Error: Improved model not found at {model_path}")
    print("Please run the train_improved.py script first to generate the improved model.")
    sys.exit(1)

# Copy the improved model to the Flask app directory
print(f"Copying improved model to Flask app directory...")
try:
    shutil.copy2(model_path, app_model_path)
    print(f"Successfully copied model to {app_model_path}")
except Exception as e:
    print(f"Error copying model: {e}")
    sys.exit(1)

# Create backup of original app.py
if os.path.exists("app.py"):
    print("Creating backup of original app.py...")
    shutil.copy2("app.py", "app.py.backup")
    print("Backup created as app.py.backup")

# Update app.py to use the improved model
with open("app.py", "r") as f:
    app_code = f.read()

# Replace the model loading line
if "plant_disease_model_1_latest.pt" in app_code:
    updated_code = app_code.replace(
        "plant_disease_model_1_latest.pt", 
        "plant_disease_model_improved.pt"
    )
    
    # Write the updated code back to app.py
    with open("app.py", "w") as f:
        f.write(updated_code)
    
    print("Successfully updated app.py to use the improved model!")
    print("\nTo use the improved model:")
    print("1. First train the model by running: python ../Model/train_improved.py")
    print("2. Then run this script to update the Flask app: python use_improved_model.py")
    print("3. Start the Flask app: python app.py")
else:
    print("Could not find model loading line in app.py. Please update manually.")