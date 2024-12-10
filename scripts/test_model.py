import joblib
from preprocessing import clean_text
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model_and_vectorizer():
    model = joblib.load('../models/spam_model.pkl')
    preprocessed_data = joblib.load('../data/preprocessed.pkl')
    vectorizer = preprocessed_data['vectorizer']
    return model, vectorizer

def get_feature_importance(message, model, vectorizer):
    # Clean and vectorize the message
    cleaned_message = clean_text(message)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get feature vector
    features = vectorizer.transform([cleaned_message])
    
    # Get coefficients from the model
    coefficients = model.coef_[0]
    
    # Get non-zero features for this message
    feature_indices = features.nonzero()[1]
    
    # Create list of (word, importance) pairs
    important_features = []
    for idx in feature_indices:
        word = feature_names[idx]
        importance = coefficients[idx] * features[0, idx]
        important_features.append((word, importance))
    
    # Sort by absolute importance
    important_features.sort(key=lambda x: abs(x[1]), reverse=True)
    return important_features[:5]  # Return top 5 features

def predict_message(message, model, vectorizer):
    # Clean the message
    cleaned_message = clean_text(message)
    
    # Vectorize the message
    features = vectorizer.transform([cleaned_message])
    
    # Predict
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    
    # Get important features
    important_features = get_feature_importance(message, model, vectorizer)
    
    return prediction, probability, important_features

def main():
    # Load model and vectorizer
    print("Loading model and vectorizer...")
    model, vectorizer = load_model_and_vectorizer()
    
    # Test messages
    test_messages = [
        # Mixed signals (legitimate-looking messages with spam-like words)
        "FREE workshop on Python programming next Tuesday at the library!",
        "URGENT: Team meeting rescheduled to 3 PM due to client emergency",
        "Limited time discount on your annual subscription - 50% off for loyal customers",
        
        # Modern spam variations
        "Your Bitcoin wallet has been credited! Log in now to claim your reward",
        "Your Netflix subscription has expired! Update payment details within 24 hours",
        "Your package delivery failed. Track your order here: bit.ly/tracking123",
        
        # Legitimate but with urgent language
        "REMINDER: Your dentist appointment is tomorrow at 2 PM",
        "Important: Please submit your project report by EOD",
        "Final call for conference registration - Early bird pricing ends today",
        
        # Subtle spam
        "Hi! I noticed your profile and think we could collaborate on some projects",
        "Your computer may have a virus! Free scan available now",
        "We've been trying to reach you about your car's extended warranty",
        
        # Normal business communication
        "The quarterly sales report is attached for your review",
        "Can we reschedule our 1:1 to next week?",
        "Please find the updated documentation in the shared drive"
    ]
    
    print("\nAnalyzing test messages:\n")
    for message in test_messages:
        prediction, probability, important_features = predict_message(message, model, vectorizer)
        is_spam = prediction == 1
        confidence = probability[1] if is_spam else probability[0]
        
        print(f"\nMessage: {message}")
        print(f"Prediction: {'SPAM' if is_spam else 'HAM'}")
        print(f"Confidence: {confidence:.2%}")
        
        print("\nTop influencing words:")
        for word, importance in important_features:
            impact = "spam" if importance > 0 else "ham"
            print(f"- '{word}' (suggests {impact}, weight: {abs(importance):.4f})")
        
        print("-" * 80)

if __name__ == "__main__":
    main()
