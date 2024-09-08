import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import os
import pickle  # Import pickle to save the tokenizer

# Load the cleaned dataset
df = pd.read_csv('data/cleaned_reviews.csv')

# Prepare labels (make sure sentiment column is properly set as 0 or 1)
df['sentiment'] = np.random.choice([0, 1], size=len(df))  # Dummy example

# Ensure that 'cleaned_review' is a string
df['cleaned_review'] = df['cleaned_review'].astype(str)

# Drop rows with empty 'cleaned_review'
df = df[df['cleaned_review'].str.strip() != '']

# Prepare tokenizer and text sequences
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['cleaned_review'])

# Save the tokenizer
with open('models/tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved successfully!")

# Convert reviews to sequences of integers
X_sequences = tokenizer.texts_to_sequences(df['cleaned_review'])

# Pad sequences to ensure uniform length
max_length = 100
X_padded = pad_sequences(X_sequences, maxlen=max_length, padding='post', truncating='post')

# Prepare labels
y = df['sentiment'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.5),
    Bidirectional(LSTM(32)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=64)

# Save the model
model.save('models/sentiment_model.h5')
print("Model saved successfully!")
