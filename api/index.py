from flask import Flask, request, jsonify
import requests
import replicate
from flask_cors import CORS
import json
import os

app = Flask(__name__)


CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173", "https://fashious-genai-hashir.netlify.app"],
        "methods": ["GET", "POST"]
    }
})

# Define your API tokens here
REPLICATE_API_TOKEN =  os.getenv("REPLICATE_API_TOKEN")
PIXELCUT_API = os.getenv("PIXELCUT_API")
PICSART_API = os.getenv("PICSART_API")

@app.route('/', methods=['GET'])
def hello_world():
    return 'Hello, World!'

@app.route('/api/transform_image', methods=['POST'])
def transform_image():

    data = request.json
    model_image_url = data.get('model_image_url')
    garment_image_url = data.get('garment_image_url')
    
    if not model_image_url or not garment_image_url:
        return jsonify({'error': 'model_image_url and garment_image_url are required'}), 400
    
    client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))
    
    try:
        output = client.run(
            "viktorfa/oot_diffusion:9f8fa4956970dde99689af7488157a30aa152e23953526a605df1d77598343d7",
            input={
                "seed": 0,
                "steps": 20,
                "model_image": model_image_url,
                "garment_image": garment_image_url,
                "guidance_scale": 2
            }
        )
        return jsonify(output)
    except Exception as e:
        return jsonify({'error': 'Failed to transform image', 'details': str(e)}), 500


@app.route('/api/backgeneratorpic', methods=['POST'])
def remove_bg():
    # Get data from the request
    data = request.json
    image_url = data.get('image_url')
    color = data.get('color')
    
    # Define the API URL and headers
    url = "https://api.picsart.io/tools/1.0/removebg"
    payload = {
        "output_type": "cutout",
        "bg_blur": "10",
        "scale": "fit",
        "auto_center": "false",
        "stroke_size": "0",
        "stroke_color": "FFFFFF",
        "stroke_opacity": "100",
        "format": "JPG",
        "image_url": image_url,
        "bg_color": color
    }
    headers = {
        "accept": "application/json",
        "X-Picsart-API-Key": PICSART_API  # Replace with your actual API key
    }

    # Make the POST request
    response = requests.post(url, data=payload, headers=headers)
    
    # Return the response from the API
    return jsonify(response.json())


@app.route('/api/generate-background', methods=['POST'])
def generate_background():
    # Get data from the request
    data = request.json
    image_url = data.get('image_url')
    prompt = data.get('prompt')

    # Define the API URL and headers
    url = "https://api.developer.pixelcut.ai/v1/generate-background"
    payload = json.dumps({
        "image_url": image_url,
        "image_transform": {
            "scale": 1,
            "x_center": 0.5,
            "y_center": 0.5
        },
        "scene": "",
        "prompt": prompt,
        "negative_prompt": ""
    })
    headers = {
        'Content-Type': 'application/json',
        'X-API-KEY': PIXELCUT_API  # Replace with your actual API key
    }

    # Make the POST request
    response = requests.post(url, headers=headers, data=payload)
    
    # Return the response from the API
    return jsonify(response.json())

if __name__ == '__main__':
    app.run(debug=True)
