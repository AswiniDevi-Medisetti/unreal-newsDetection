# app.py
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

app = Flask(__name__)

# Load or train your ML model
try:
    # Try to load pre-trained model
    vectorizer = joblib.load('vectorizer.pkl')
    model = joblib.load('fake_news_model.pkl')
    print("Pre-trained model loaded successfully")
except:
    # If model doesn't exist, create a simple one for demo
    print("Training a new model...")
    
    # Sample data (in a real application, you'd use a proper dataset)
    data = {
        'text': [
            "Breaking: Scientists discover revolutionary new energy source that will change the world forever",
            "The government announces new policies to improve healthcare services nationwide",
            "Aliens have been confirmed to exist and are living among us in secret",
            "Local community raises funds for new park renovation project",
            "Celebrity claims to have found the secret to eternal youth in rare fruit",
            "Economic indicators show steady growth in the manufacturing sector",
            "Politician makes outrageous claim about opponent without evidence",
            "New study reveals benefits of regular exercise for mental health",
            "Miracle cure for all diseases discovered by high school student",
            "Weather forecast predicts mild temperatures for the weekend"
        ],
        'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 for fake, 0 for real
    }
    
    df = pd.DataFrame(data)
    
    # Simple text preprocessing
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label']
    
    # Train model
    model = LogisticRegression()
    model.fit(X, y)
    
    # Save model for future use
    joblib.dump(vectorizer, 'vectorizer.pkl')
    joblib.dump(model, 'fake_news_model.pkl')
    print("Model trained and saved successfully")

def extract_text_from_url(url):
    """Extract text content from a URL"""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract text from common article tags
        article_text = ""
        for tag in ['article', 'main', '.article-content', '.post-content']:
            elements = soup.select(tag)
            for element in elements:
                article_text += element.get_text() + " "
        
        # If no specific article content found, get all text
        if not article_text.strip():
            article_text = soup.get_text()
            
        return article_text[:5000]  # Limit text length
    except Exception as e:
        print(f"Error extracting text from URL: {e}")
        return ""

def analyze_news_content(text):
    """Analyze news content and return prediction with confidence"""
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Vectorize
    text_vector = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    confidence = np.max(model.predict_proba(text_vector)) * 100
    
    # Additional analysis (simplified for demo)
    analysis_details = {
        'source_credibility': min(90, max(50, 100 - confidence/2)),
        'linguistic_analysis': min(95, max(60, confidence - 5)),
        'fact_checking': min(90, max(55, 100 - confidence/3)),
        'sentiment_analysis': min(85, max(50, confidence - 10))
    }
    
    return {
        'is_fake': bool(prediction),
        'confidence': round(confidence, 2),
        'analysis_details': analysis_details
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    
    text = data.get('text', '')
    url = data.get('url', '')
    
    # If URL provided, extract text from it
    if url and not text:
        text = extract_text_from_url(url)
        if not text:
            return jsonify({
                'error': 'Could not extract text from the provided URL'
            }), 400
    
    if not text:
        return jsonify({
            'error': 'No text content provided for analysis'
        }), 400
    
    # Analyze the content
    try:
        result = analyze_news_content(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'error': f'Analysis failed: {str(e)}'
        }), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Endpoint for analyzing multiple news items at once"""
    data = request.json
    texts = data.get('texts', [])
    
    results = []
    for text in texts:
        if text.strip():
            result = analyze_news_content(text)
            results.append(result)
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)