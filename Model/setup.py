import subprocess
import sys
import os

print("Plant Disease Detection - Improved Model Setup")
print("-" * 50)

# Check if pip is available
try:
    subprocess.check_call([sys.executable, "-m", "pip", "--version"])
    print("pip is installed. Proceeding with setup...")
except:
    print("Error: pip is not installed or not in PATH. Please install pip first.")
    sys.exit(1)

# Install requirements
print("\nInstalling required packages...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("Successfully installed required packages!")
except Exception as e:
    print(f"Error installing packages: {e}")
    sys.exit(1)

# Check if Dataset directory exists
if not os.path.exists("Dataset"):
    print("\nWARNING: Dataset directory not found!")
    print("Please download the Plant Village dataset from:")
    print("https://data.mendeley.com/datasets/tywbtsjrjv/1")
    print("Extract it and ensure there's a 'Dataset' folder in the Model directory.")
else:
    print("\nDataset directory found!")

print("\nSetup completed successfully!")
print("\nTo train the improved model, run:")
print("python train_improved.py")
print("\nAfter training, to update the Flask app with the new model, run:")
print("python ../Flask\ Deployed\ App/use_improved_model.py")