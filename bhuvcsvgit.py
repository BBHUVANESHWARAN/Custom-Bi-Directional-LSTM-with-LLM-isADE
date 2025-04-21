import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

class IntentClassifier:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.tokenizer = Tokenizer()
        self.max_len = None
        self.labels = None
        self.model = None
    def load_data(self):
        try:
            df = pd.read_csv(self.csv_path)
            contents = df['content'].tolist()
            intents = df['intent'].tolist()
            return contents, intents
        except Exception as e:
            print(f"Failed to read the CSV file: {e}")
            return [], []

    def preprocess_data(self, contents, intents):
        self.tokenizer.fit_on_texts(contents)
        X = self.tokenizer.texts_to_sequences(contents)
        self.max_len = max([len(x) for x in X])
        X = pad_sequences(X, maxlen=self.max_len)

        self.labels = list(set(intents))
        label_to_index = {label: i for i, label in enumerate(self.labels)}
        y = np.array([label_to_index[label] for label in intents])
        return X, y

    def build_model(self, output_dim):
        self.model = Sequential([
            Embedding(input_dim=len(self.tokenizer.word_index)+1, output_dim=output_dim),
            Bidirectional(LSTM(64)),
            Dense(len(self.labels), activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, X, y, epochs=50, batch_size=32):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    def save_model(self, file_path):
        self.model.save(file_path)

    def predict_intent(self, user_input):
        try:
            X_input = self.tokenizer.texts_to_sequences([user_input])
            X_input = pad_sequences(X_input, maxlen=self.max_len)
            prediction = self.model.predict(X_input)
            predicted_label_index = np.argmax(prediction)
            return self.labels[predicted_label_index]
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None
import os

def save_model(self, file_path):
    # Delete the existing model file if it exists
    if os.path.exists(file_path):
        os.remove(file_path)

    try:
        self.model.save(file_path)
        print("Model saved successfully.")
    except Exception as e:
        print(f"Error saving the model: {e}")
# Usage
if __name__ == "__main__":
    classifier = IntentClassifier('E:\\BHUVANESH\\WORKOUT\\GP_Mean_covariance\\BiLSTM\\intents_content.csv')
    contents, intents = classifier.load_data()
    X, y = classifier.preprocess_data(contents, intents)
    classifier.build_model(128)
    classifier.train_model(X, y)
    # Using forward slashes
    classifier.save_model(r'E:/BHUVANESH/WORKOUT/GP_Mean_covariance/BiLSTM/intent_model.h5')

    # Using double backslashes
    classifier.save_model('E:\\BHUVANESH\\WORKOUT\\GP_Mean_covariance\\BiLSTM\\intent_model.h5')
    user_input = "i need solutions for screens on the terminals via Cloud"
    predicted_intent = classifier.predict_intent(user_input)
    print("Predicted Intent:", predicted_intent)
