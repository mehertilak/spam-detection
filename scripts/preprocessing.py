import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import joblib
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print("\nSample of first few rows:")
    print(df.head())
    return df

# Text cleaning
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Vectorization
def vectorize_text(data):
    vectorizer = TfidfVectorizer(max_features=1000)
    features = vectorizer.fit_transform(data)
    print(f"\nFeature matrix shape: {features.shape}")
    return features, vectorizer

# Save preprocessed data
def save_preprocessed_data(features, labels, vectorizer, file_path):
    joblib.dump({
        'features': features,
        'labels': labels,
        'vectorizer': vectorizer
    }, file_path)

if __name__ == "__main__":
    # Load and preprocess data
    data_path = "c:/Users/Tilak/AppData/Local/Temp/d6d61899-bed3-478c-9ea4-c1081ae662da_archive (1).zip.2da/spam.csv"
    data = load_data(data_path)
    
    # Ensure we have the correct column names for text and label
    text_column = 'v2' if 'v2' in data.columns else 'text'  # Adjust based on actual column name
    label_column = 'v1' if 'v1' in data.columns else 'label'  # Adjust based on actual column name
    
    # Convert labels to numeric
    data[label_column] = data[label_column].map({'spam': 1, 'ham': 0})
    
    print(f"\nLabel distribution:")
    print(data[label_column].value_counts())
    
    # Clean and preprocess text
    print("\nCleaning text...")
    data['clean_text'] = data[text_column].apply(clean_text)
    
    # Vectorize text
    print("\nVectorizing text...")
    features, vectorizer = vectorize_text(data['clean_text'])
    
    # Save preprocessed data
    print("\nSaving preprocessed data...")
    save_preprocessed_data(features, data[label_column], vectorizer, '../data/preprocessed.pkl')
    print("Preprocessing complete!")
