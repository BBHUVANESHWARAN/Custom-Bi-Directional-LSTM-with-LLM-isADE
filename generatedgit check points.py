import os
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

class IntentClassifier:
    def __init__(self, csv_path, checkpoint_dir, output_dim=128):
        self.csv_path = csv_path
        self.tokenizer = Tokenizer()
        self.max_len = None
        self.labels = None
        self.model = None
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.output_dim = output_dim
        # Ensure the Checkpoint object is created here
        self.checkpoint = tf.train.Checkpoint()

    def build_model(self):
        if self.labels is None:
            raise ValueError("Labels are not set. Cannot build model without label information.")
        
        self.model = Sequential([
            Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=128),
            Bidirectional(LSTM(64)),
            Dense(len(self.labels), activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # Bind the model to the existing checkpoint object
        self.checkpoint.model = self.model
        
    def save_model(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        print("Model checkpoint saved.")

    def load_model(self):
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest:
            if self.model is None:
                self.build_model()  # Ensure model is built if it's not already
            self.checkpoint.restore(latest)
            print("Model restored from checkpoint.")
        

    def train_model(self, X, y, epochs=100, batch_size=32):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        self.save_model()

    def predict_intent(self, user_input):
        X_input = self.tokenizer.texts_to_sequences([user_input])
        X_input = pad_sequences(X_input, maxlen=self.max_len)
        prediction = self.model.predict(X_input)
        predicted_label_index = np.argmax(prediction)
        return self.labels[predicted_label_index]


    def preprocess_data(self, contents, intents):
            try:
                self.tokenizer.fit_on_texts(contents)
                X = self.tokenizer.texts_to_sequences(contents)
                self.max_len = max([len(x) for x in X])
                X = pad_sequences(X, maxlen=self.max_len)

                self.labels = list(set(intents))
                label_to_index = {label: i for i, label in enumerate(self.labels)}
                y = np.array([label_to_index[label] for label in intents])
                return X, y
            except Exception as e:
                print(f"Failed to preprocess data: {e}")
                return [], []
    def predict_intent(self, user_input):
        X_input = self.tokenizer.texts_to_sequences([user_input])
        X_input = pad_sequences(X_input, maxlen=self.max_len)
        prediction = self.model.predict(X_input)
        predicted_label_index = np.argmax(prediction)
        return self.labels[predicted_label_index]
    
    def load_data(self):
        try:
            df = pd.read_csv(self.csv_path)
            contents = df['Generated Question'].tolist()
            intents = df['Intent'].tolist()
            return contents, intents
        except Exception as e:
            print(f"Failed to read the CSV file: {e}")
            return [], []
    
def main():
    base_path = r'E:\BHUVANESH\WORKOUT\GP_Mean_covariance\BiLSTM\Intent'
    checkpoint_dir = os.path.join(base_path, 'model')
    questions_csv_path = os.path.join(base_path, 'generated_questions.csv')

    classifier = IntentClassifier(questions_csv_path, checkpoint_dir)
    contents, intents = classifier.load_data()
    if contents and intents:  # Ensure data is loaded
        X, y = classifier.preprocess_data(contents, intents)
    
    if tf.train.latest_checkpoint(classifier.checkpoint_dir) is None:
        classifier.build_model()  # Ensure model is built after labels are available
        classifier.train_model(X, y)
    else:
        classifier.build_model()
        classifier.train_model(X, y)
        classifier.load_model()  # Load model must be called after ensuring data is loaded and labels are set

    while True:
        user_input = input("Enter your question: ")
        predicted_intent = classifier.predict_intent(user_input)
        print("Predicted Intent:", predicted_intent)

if __name__ == "__main__":
    main()


