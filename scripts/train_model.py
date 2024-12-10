import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

# Load preprocessed data
def load_preprocessed_data(file_path):
    data = joblib.load(file_path)
    print(f"Loaded feature matrix shape: {data['features'].shape}")
    print(f"Number of samples: {len(data['labels'])}")
    return data

# Train model
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='weighted')
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, predictions))
    
    return accuracy, f1

if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    data = load_preprocessed_data('../data/preprocessed.pkl')
    
    # Split data
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        data['features'], data['labels'],
        test_size=0.2,
        random_state=42,
        stratify=data['labels']  # Ensure balanced split
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    
    # Train model
    print("\nTraining model...")
    model = train_model(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, f1 = evaluate_model(model, X_test, y_test)
    print(f"\nFinal Metrics:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1 Score: {f1:.3f}")
    
    # Save model
    print("\nSaving model...")
    joblib.dump(model, '../models/spam_model.pkl')
    print("Training complete!")
