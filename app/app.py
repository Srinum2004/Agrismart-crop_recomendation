# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# # Load the model and preprocessing objects
# model = joblib.load(r'C:\Users\HP\OneDrive\Desktop\6th sem clg\SSP Project\Crop recomendation system\Notebook\crop_recommender_model.pkl')
# scaler = joblib.load(r'C:\Users\HP\OneDrive\Desktop\6th sem clg\SSP Project\Crop recomendation system\Notebook\scaler.pkl')
# encoder = joblib.load(r'C:\Users\HP\OneDrive\Desktop\6th sem clg\SSP Project\Crop recomendation system\Notebook\encoder.pkl')

# Load the model and preprocessing objects
model = joblib.load(r'C:\Users\HP\OneDrive\Desktop\6th sem clg\SSP Project\Crop recomendation system\models\crop_model_nav.pkl')
scaler = joblib.load(r'C:\Users\HP\OneDrive\Desktop\6th sem clg\SSP Project\Crop recomendation system\models\scaler_nav.pkl')
encoder = joblib.load(r'C:\Users\HP\OneDrive\Desktop\6th sem clg\SSP Project\Crop recomendation system\models\encoder_nav.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array([[
        data['N'], data['P'], data['K'], 
        data['temperature'], data['humidity'], 
        data['ph'], data['rainfall']
    ]])
    
    # Preprocess
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)
    crop = encoder.inverse_transform(prediction)
    
    return jsonify({'crop': crop[0]})

if __name__ == '__main__':
    app.run(debug=True)