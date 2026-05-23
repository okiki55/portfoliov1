"""
Sentiment Analyzer
==================
Example NLP project demonstrating text sentiment analysis.

This is a DEMO implementation using keyword-based analysis.
Replace this with your actual trained model for real predictions.
"""

import os
import re

# Uncomment for real model
# import pickle
# import numpy as np


def load_model():
    """
    Load your trained sentiment model.
    
    Example with sklearn:
    -----------------------
    import pickle
    model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer
    """
    return None


def predict(data):
    """
    Analyze sentiment of input text.
    
    Args:
        data (dict): Dictionary containing:
                     - 'text': The text to analyze
                     - 'language': The language (optional)
                     
    Returns:
        dict: Sentiment prediction with confidence scores
    """
    
    text = data.get('text', '')
    language = data.get('language', 'English')
    
    if not text or not text.strip():
        return {'error': 'Please provide text to analyze'}
    
    # Clean text
    text = text.lower().strip()
    
    # ============================================
    # DEMO: Simple keyword-based sentiment
    # REPLACE THIS with your actual model prediction
    # ============================================
    
    # Positive keywords
    positive_words = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'love', 'loved', 'loving', 'like', 'liked', 'best', 'happy', 'joy',
        'beautiful', 'perfect', 'awesome', 'brilliant', 'outstanding',
        'recommend', 'recommended', 'satisfied', 'thank', 'thanks', 'pleased',
        'impressive', 'exceptional', 'superb', 'delightful', 'positive'
    ]
    
    # Negative keywords
    negative_words = [
        'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'hated',
        'dislike', 'disappointed', 'disappointing', 'poor', 'waste', 'useless',
        'broken', 'fail', 'failed', 'failure', 'wrong', 'problem', 'issue',
        'angry', 'upset', 'frustrated', 'annoyed', 'sad', 'unhappy',
        'never', 'not', 'don\'t', 'doesn\'t', 'didn\'t', 'won\'t', 'wouldn\'t',
        'boring', 'slow', 'expensive', 'overpriced', 'negative'
    ]
    
    # Count occurrences
    words = re.findall(r'\b\w+\b', text)
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    # Check for negation patterns
    negation_patterns = ['not good', 'not great', 'not recommend', 'don\'t like', 'didn\'t like']
    for pattern in negation_patterns:
        if pattern in text:
            negative_count += 2
            positive_count = max(0, positive_count - 1)
    
    # Calculate sentiment scores
    total = positive_count + negative_count + 1  # +1 to avoid division by zero
    
    if positive_count > negative_count:
        # Positive sentiment
        pos_score = 0.5 + (positive_count / (total * 2))
        neg_score = negative_count / (total * 3)
        neu_score = 1 - pos_score - neg_score
        sentiment = 'Positive'
        confidence = min(0.95, pos_score)
    elif negative_count > positive_count:
        # Negative sentiment
        neg_score = 0.5 + (negative_count / (total * 2))
        pos_score = positive_count / (total * 3)
        neu_score = 1 - pos_score - neg_score
        sentiment = 'Negative'
        confidence = min(0.95, neg_score)
    else:
        # Neutral sentiment
        neu_score = 0.6
        pos_score = 0.2
        neg_score = 0.2
        sentiment = 'Neutral'
        confidence = 0.6
    
    # Ensure scores sum to 1
    total_score = pos_score + neg_score + neu_score
    pos_score /= total_score
    neg_score /= total_score
    neu_score /= total_score
    
    # ============================================
    # If you have a real model, use it like this:
    # ============================================
    # 
    # model, vectorizer = load_model()
    # 
    # # Vectorize text
    # text_vector = vectorizer.transform([text])
    # 
    # # Predict
    # prediction = model.predict(text_vector)[0]
    # probabilities = model.predict_proba(text_vector)[0]
    # 
    # classes = ['Negative', 'Neutral', 'Positive']
    # sentiment = classes[prediction]
    # confidence = float(max(probabilities))
    
    return {
        'result': sentiment,
        'confidence': confidence,
        'details': {
            'positive_score': f"{pos_score:.1%}",
            'negative_score': f"{neg_score:.1%}",
            'neutral_score': f"{neu_score:.1%}",
            'word_count': len(words),
            'language': language
        },
        'chart_data': {
            'type': 'doughnut',
            'label': 'Sentiment Distribution',
            'labels': ['Positive', 'Negative', 'Neutral'],
            'values': [
                round(pos_score * 100, 1),
                round(neg_score * 100, 1),
                round(neu_score * 100, 1)
            ]
        }
    }
