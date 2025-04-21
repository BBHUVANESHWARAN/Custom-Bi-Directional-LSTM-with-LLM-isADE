import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

class IntentPredictor:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = Tokenizer()
        self.labels = None
        self.model = None
        self.max_len = None

    def load_model_and_tokenizer(self):
        try:
            # Load tokenizer's word index from JSON
            tokenizer_path = os.path.join(self.checkpoint_dir, 'tokenizer.json')
            with open(tokenizer_path, 'r', encoding='utf-8') as f:
                word_index = json.load(f)
            self.tokenizer.word_index = word_index
            
            # Load labels
            labels_path = os.path.join(self.checkpoint_dir, 'labels.json')
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.labels = json.load(f)

            # Build model architecture
            self.model = self.build_model()
            
            # Load model weights using TensorFlow checkpoints
            checkpoint = tf.train.Checkpoint(model=self.model)
            latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
            if latest_checkpoint:
                checkpoint.restore(latest_checkpoint).expect_partial()
                print("Model weights restored successfully.")
            else:
                print("No checkpoint found. Model weights not restored.")

            return True
        except Exception as e:
            print(f"Failed to load model and tokenizer: {e}")
            return False

    def build_model(self):
        # Define model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=128),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(len(self.labels), activation='softmax')
        ])
        return model

    def preprocess_input(self, text):
        try:
            # Tokenize input text
            sequence = self.tokenizer.texts_to_sequences([text])
            # Pad sequences
            padded_sequence = pad_sequences(sequence, maxlen=self.max_len)
            return padded_sequence
        except Exception as e:
            print(f"Error preprocessing input: {e}")
            return None

    def predict_intent(self, text):
        try:
            if self.model is None or self.tokenizer is None:
                print("Model and tokenizer not loaded.")
                return None

            # Preprocess input text
            input_sequence = self.preprocess_input(text)
            if input_sequence is None:
                return None

            # Make prediction
            predictions = self.model.predict(input_sequence)
            predicted_index = np.argmax(predictions)
            
            # Convert predicted index to int
            predicted_intent = self.labels[int(predicted_index)]  # Convert index to int before indexing labels
            return predicted_intent
        except Exception as e:
            print(f"Error predicting intent: {e}")
            return None

    def main():
        try:
            checkpoint_dir = os.path.join('apps', 'static', 'data', 'model', 'checkpoints')
            predictor = IntentPredictor(checkpoint_dir)
            if predictor.load_model_and_tokenizer():
                while True:
                    user_input = input("Enter your question: ")
                    if user_input.lower() == 'exit':
                        break
                    predicted_intent = predictor.predict_intent(user_input)
                    if predicted_intent:
                        print(f"Predicted intent: {predicted_intent}")
                    else:
                        print("Failed to predict intent.")
        except KeyboardInterrupt:
            print("Prediction stopped by user.")


IntentPredictor.main()
