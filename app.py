from flask import Flask, request, jsonify
import torch
import numpy as np
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load your PyTorch model
model = torch.load('model.pt')
model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        rows = data.get('rows', [])
        
        predictions = []
        for row in rows:
            # Extract band values
            band1 = float(row.get('band1', 0))
            band2 = float(row.get('band2', 0))
            band3 = float(row.get('band3', 0))
            band4 = float(row.get('band4', 0))
            
            # Prepare input tensor
            input_tensor = torch.tensor([[band1, band2, band3, band4]], dtype=torch.float32)
            
            # Run model
            with torch.no_grad():
                output = model(input_tensor)
                salinity = float(output[0].item())
            
            predictions.append({
                'latitude': row.get('latitude'),
                'longitude': row.get('longitude'),
                'band1': band1,
                'band2': band2,
                'band3': band3,
                'band4': band4,
                'salinity': salinity
            })
        
        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)