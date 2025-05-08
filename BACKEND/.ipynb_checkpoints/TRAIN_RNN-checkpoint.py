import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import joblib
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Embedding, SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# Load the data
true = pd.read_csv('True.csv')
fake = pd.read_csv('Fake.csv')

# Assign labels
true['label'] = 1
fake['label'] = 0
print("Concatenate dataframes")
# Concatenate dataframes
frames = [true.loc[0:5000], fake.loc[0:5000]]
df = pd.concat(frames).sample(frac=1).reset_index(drop=True)


print("Preprocessing")
# Preprocessing
messages = df.copy()
messages = messages.dropna()
ps = PorterStemmer()
corpus = []

for i in range(len(messages)):
    print(f"Corpus : {i} out of {len(messages)}")
    review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# Tokenization and padding
max_words = 5000
max_len = 150
tokenizer = Tokenizer(num_words=max_words, split=' ')
tokenizer.fit_on_texts(corpus)
X = tokenizer.texts_to_sequences(corpus)
X = pad_sequences(X, maxlen=max_len)

# Train-test split
y = messages['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Define RNN model
embed_dim = 128
rnn_out = 196
model = Sequential()
model.add(Embedding(max_words, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(SimpleRNN(rnn_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
batch_size = 32
history = model.fit(X_train, y_train, epochs=10, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)

# Plot accuracy and loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Save the model and tokenizer
model.save('rnn_model.h5')
joblib.dump(tokenizer, 'tokenizer.pkl')
