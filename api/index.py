from flask import Flask, request, jsonify
import requests
import replicate
from flask_cors import CORS

app = Flask(__name__)


CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST"]
    }
})

# Define your API tokens here
BRIA_API_TOKEN = '568187d58ec14982b5959c71c48fda74'
REPLICATE_API_TOKEN = 'r8_XtM5nuur66PdgoZkRnxZ5M45vaZcUUW3okLjY'

@app.route('/api/replace_background', methods=['POST'])
def replace_background():
    data = request.json
    bg_prompt = data.get('bg_prompt')
    image_url = data.get('image_url')
    
    if not bg_prompt or not image_url:
        return jsonify({'error': 'bg_prompt and image_url are required'}), 400
    
    url = "https://engine.prod.bria-api.com/v1/background/replace"
    payload = {
        "bg_prompt": bg_prompt,
        "num_results": 2,
        "sync": True,
        "image_url": image_url
    }
    headers = {
        "Content-Type": "application/json",
        "api_token": BRIA_API_TOKEN
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code != 200:
        return jsonify({'error': 'Failed to replace background', 'details': response.text}), response.status_code
    
    data = response.json()
    return jsonify(data)

@app.route('/api/transform_image', methods=['POST'])
def transform_image():

    data = request.json
    model_image_url = data.get('model_image_url')
    garment_image_url = data.get('garment_image_url')
    
    if not model_image_url or not garment_image_url:
        return jsonify({'error': 'model_image_url and garment_image_url are required'}), 400
    
    client = replicate.Client(api_token=REPLICATE_API_TOKEN)
    
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

if __name__ == '__main__':
    app.run(debug=True)
