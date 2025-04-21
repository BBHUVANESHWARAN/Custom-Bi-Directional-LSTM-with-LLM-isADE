import warnings
warnings.filterwarnings("ignore")
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class ini_int_pred:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.tokenizer = Tokenizer()
        self.labels = None
        self.model = None
        self.max_len = 100  # Set this to your training max length or a default value

    #
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

            # Ensure labels and tokenizer are loaded before building model
            if self.labels and word_index:
                self.model = self.build_model()
                # Load model weights using TensorFlow checkpoints
                checkpoint = tf.train.Checkpoint(model=self.model)
                latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
                if latest_checkpoint:
                    checkpoint.restore(latest_checkpoint).expect_partial()
                    print("Model weights restored successfully.")
                else:
                    print("No checkpoint found. Model weights not restored.")
            else:
                print("Failed to load tokenizer or labels.")

            return True
        except Exception as e:
            print(f"Failed to load model and tokenizer: {e}")
            return False

    def build_model(self):
        # Define model architecture
        model = Sequential([
            Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=128),
            Bidirectional(LSTM(64)),
            Dense(len(self.labels), activation='softmax')
        ])
        return model

    def preprocess_input(self, text):
        # Tokenize and pad input text
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len)
        return padded_sequence

    def predict_intent(self, text):
        if self.model is None or self.tokenizer is None or not self.labels:
            print("Model and tokenizer not loaded.")
            return None
        threshold=0.5
        input_sequence = self.preprocess_input(text)
        predictions = self.model.predict(input_sequence)
        predicted_index = np.argmax(predictions)
        predicted_confidence = np.max(predictions)
        predicted_intent = self.labels[predicted_index]
        if predicted_confidence >= threshold:
            return predicted_intent, predicted_confidence
        else:
            return "Uncertain", predicted_confidence
    

    def main(user_qry):
        user_qry = user_qry.strip()  # Remove leading and trailing whitespace
        if not user_qry:
            qry="user_queries_not_identify"
            return qry
        else:
            checkpoint_dir = os.path.join('BiLSTM','intent', 'model', 'nn_model')
            predictor = ini_int_pred(checkpoint_dir)
            if predictor.load_model_and_tokenizer():
                predicted_intent,predicted_confidence = predictor.predict_intent(user_qry)
                threshold = 0.7  # Set your desired threshold here
                if predicted_confidence >= threshold:
                    # print("Predicted Intent:", predicted_intent)
                    return predicted_intent
                else:
                    # print("Uncertain Prediction")
                    uncertain="uncertain_prediction"
                    return uncertain
                # return predicted_intent,predicted_confidence
            

user_qry = "Which of the following is a feature-rich Kiosk that support multiple application?"
predicted_intent= ini_int_pred.main(user_qry)
print(predicted_intent)
