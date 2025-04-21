import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding
from sklearn.model_selection import train_test_split

# Load JSON dataset
with open(r'E:\BHUVANESH\WORKOUT\GP_Mean_covariance\BiLSTM\json_data_extra.json', 'r') as file:
    data = json.load(file)

# Flatten the nested list of dictionaries and extract text content and intents
contents = []
intents = []
for item in data['intents']:
    if isinstance(item['content'], list):
        for subitem in item['content']:
            contents.append(subitem['content'])
            intents.append(item['intent'])
    else:
        contents.append(item['content'])
        intents.append(item['intent'])

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(contents)
X = tokenizer.texts_to_sequences(contents)

# Pad sequences to ensure uniform input size
max_len = max([len(x) for x in X])
X = pad_sequences(X, maxlen=max_len)

# Convert intents to numerical labels
labels = list(set(intents))
label_to_index = {label: i for i, label in enumerate(labels)}
y = np.array([label_to_index[label] for label in intents])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Bidirectional LSTM model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128),  # Remove input_length here
    Bidirectional(LSTM(64)),
    Dense(len(labels), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Function to predict intent for user input
def predict_intent(user_input):
    # Preprocess the user input
    user_input = [user_input]

    # Tokenize the text data
    X_input = tokenizer.texts_to_sequences(user_input)

    # Pad sequences to ensure uniform input size
    X_input = pad_sequences(X_input, maxlen=max_len)

    # Predict the intent
    prediction = model.predict(X_input)

    # Map the predicted label back to the original intent
    predicted_label_index = np.argmax(prediction)
    predicted_intent = labels[predicted_label_index]

    return predicted_intent

# Example usage:
user_input = "Below are some of the Banks we take pride in calling our clients."
predicted_intent = predict_intent(user_input)
print("Predicted Intent:", predicted_intent)
