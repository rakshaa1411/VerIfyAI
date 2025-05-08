import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Load the data
true = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

# Assign labels
true['label'] = 1
fake['label'] = 0

# Concatenate dataframes
frames = [true.loc[0:5000], fake.loc[0:5000]]
df = pd.concat(frames).sample(frac=1).reset_index(drop=True)

# Preprocessing
messages = df.copy()
messages = messages.dropna()
ps = PorterStemmer()
corpus = []
for i in range(len(messages)):
    print(f" CORPUS {i} out of {len(messages)}")
    review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(corpus).toarray()

# Train-test split
y = messages['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


import pickle

# Save the model and vectorizer using pickle
with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump(knn_model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

    
# Save the model and vectorizer
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
