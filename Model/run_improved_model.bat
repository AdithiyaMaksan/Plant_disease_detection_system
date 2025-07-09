@echo off
echo Plant Disease Detection - Improved Model Runner
echo ================================================
echo.

echo This script will help you train and use the improved plant disease detection model
echo that can accurately predict diseases from any image source, including Google Images.
echo.

set /p choice=Do you want to train the model or use it with an image? (train/use): 

if /i "%choice%"=="train" (
    echo.
    echo Starting model training...
    echo This may take some time depending on your hardware.
    echo.
    python train_improved_enhanced.py
    
    echo.
    echo Training complete! The model is now ready to use with Google images.
    echo To update the Flask app with the new model, run: python update_flask_app.py
    echo.
    pause
) else if /i "%choice%"=="use" (
    echo.
    set /p image_path=Enter the path to your image (e.g., ../test_images/apple_scab.JPG): 
    
    if exist "%image_path%" (
        echo.
        echo Analyzing image: %image_path%
        echo.
        python google_image_demo.py "%image_path%"
    ) else (
        echo.
        echo Error: Image file not found at %image_path%
        echo Please check the path and try again.
    )
    
    echo.
    pause
) else (
    echo.
    echo Invalid choice. Please run the script again and enter 'train' or 'use'.
    echo.
    pause
)