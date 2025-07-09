import os
from flask import Flask, redirect, render_template, request
from PIL import Image
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
import requests
import json
import urllib.parse
import re


disease_info = pd.read_csv('disease_info.csv' , encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv',encoding='cp1252')

model = CNN.CNN(39)    
model.load_state_dict(torch.load("plant_disease_model_1_latest.pt"))
model.eval()

def prediction(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image)
    input_data = input_data.view((-1, 3, 224, 224))
    output = model(input_data)
    output_np = output.detach().numpy()
    index = np.argmax(output_np)
    
    # Calculate prediction accuracy/confidence
    softmax_output = torch.nn.functional.softmax(output, dim=1).detach().numpy()[0]
    accuracy = softmax_output[index] * 100  # Convert to percentage
    
    return index, accuracy


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

def get_google_image(query, fallback_url):
    try:
        # Format the search query
        search_query = urllib.parse.quote(f"{query} plant disease high quality")
        
        # Use a simple approach to get an image URL
        # This is a simplified implementation - in production, you would use a proper API
        search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(search_url, headers=headers)
        
        # Extract image URLs using regex
        if response.status_code == 200:
            # Look for image URLs in the response
            img_urls = re.findall(r'"(https://[^"]+\.(?:jpg|jpeg|png))"', response.text)
            if img_urls:
                # Return the first valid image URL
                for url in img_urls:
                    if 'http' in url and ('jpg' in url.lower() or 'jpeg' in url.lower() or 'png' in url.lower()):
                        return url
        
        # If we couldn't get a Google image, return the fallback URL
        return fallback_url
    except Exception as e:
        print(f"Error fetching Google image: {e}")
        return fallback_url

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        image = request.files['image']
        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred, accuracy = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        
        # Get the fallback image URL from the CSV
        fallback_image_url = disease_info['image_url'][pred]
        
        # Try to get a better reference image from Google
        google_image_url = get_google_image(title, fallback_image_url)
        
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        
        # Pass just the filename to the template
        uploaded_image_filename = filename
        return render_template('submit.html', title=title, desc=description, prevent=prevent, 
                               image_url=google_image_url, pred=pred, sname=supplement_name, 
                               simage=supplement_image_url, buy_link=supplement_buy_link, 
                               accuracy=round(accuracy, 2), uploaded_image=uploaded_image_filename)

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', supplement_image = list(supplement_info['supplement image']),
                           supplement_name = list(supplement_info['supplement name']), disease = list(disease_info['disease_name']), buy = list(supplement_info['buy link']))

if __name__ == '__main__':
    app.run(debug=True)
