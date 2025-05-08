import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('rnn_model.h5')
tokenizer = joblib.load('tokenizer.pkl')

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
    sequence = tokenizer.texts_to_sequences([preprocessed_text])
    padded_sequence = pad_sequences(sequence, maxlen=150)
    prediction = model.predict(padded_sequence)
    if prediction[0][0] > prediction[0][1]:
        return "Fake"
    else:
        return "Real"

# Get input from user
input_text = input("Enter the news text: ")
prediction = predict_fake_news(input_text)
print("Prediction:", prediction)
