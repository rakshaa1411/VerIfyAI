import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load model and vectorizer
knn_model = joblib.load('knn_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocess input text
def preprocess_input(text):
    ps = PorterStemmer()
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review

# Predict function
def predict_fake_news(text):
    preprocessed_text = preprocess_input(text)
    vectorized_text = vectorizer.transform([preprocessed_text]).toarray()
    prediction = knn_model.predict(vectorized_text)
    if prediction[0] == 1:
        return "Real"
    else:
        return "Fake"

# Get input from user
input_text = input("Enter the news text: ")
prediction = predict_fake_news(input_text)
print("Prediction:", prediction)
