from flask import Flask, request, jsonify, send_from_directory
from preprocessing import clean_text
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model and vectorizer
model_path = os.path.join('..', 'models', 'spam_model.pkl')
preprocessed_path = os.path.join('..', 'data', 'preprocessed.pkl')

model = joblib.load(model_path)
preprocessed_data = joblib.load(preprocessed_path)
vectorizer = preprocessed_data['vectorizer']

@app.route('/')
def serve_frontend():
    return send_from_directory('../static', 'index.html')

@app.route('/predict', methods=['POST'])
def predict_spam():
    try:
        # Get message from request
        data = request.get_json()
        if 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
        
        message = data['message']
        
        # Clean and vectorize the message
        cleaned_message = clean_text(message)
        features = vectorizer.transform([cleaned_message])
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Get confidence score
        confidence = probability[1] if prediction == 1 else probability[0]
        
        # Prepare response
        response = {
            'message': message,
            'is_spam': bool(prediction),
            'confidence': float(confidence),
            'prediction': 'SPAM' if prediction == 1 else 'HAM',
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Spam Detection Service...")
    print("Access the web interface at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000)
