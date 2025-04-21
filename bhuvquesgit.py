import warnings
warnings.filterwarnings("ignore")
import csv
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration

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
            contents = df['Generated Question'].tolist()
            intents = df['Intent'].tolist()
            return contents, intents
        except Exception as e:
            print(f"Failed to read the CSV file: {e}")
            return [], []

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

    def build_model(self, output_dim):
        try:
            self.model = Sequential([
                Embedding(input_dim=len(self.tokenizer.word_index)+1, output_dim=output_dim),
                Bidirectional(LSTM(64)),
                Dense(len(self.labels), activation='softmax')
            ])
            self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        except Exception as e:
            print(f"Failed to build the model: {e}")

    def train_model(self, X, y, epochs=50, batch_size=32):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        except Exception as e:
            print(f"Failed to train the model: {e}")

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

    def save_model(self, file_path):
        try:
            # Delete the existing model file if it exists
            if os.path.exists(file_path):
                os.remove(file_path)

            self.model.save(file_path)
            print("Model saved successfully.")
        except Exception as e:
            print(f"Error saving the model: {e}")

def generate_questions(tokenizer,model,content):
    try:


        input_text = f"Based on the following information, generate ten questions whose answers can be directly found in the text: \"{content}\" Ensure each question corresponds to specific details or facts mentioned."
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        outputs = model.generate(
            input_ids,
            max_length=200,
            num_return_sequences=10,
            num_beams=10,
            temperature=0.9,
            no_repeat_ngram_size=1
        )
        generated_questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return generated_questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

if __name__ == "__main__":
    try:
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
        # Generate questions from content data and save to CSV
        with open('E:\BHUVANESH\WORKOUT\GP_Mean_covariance\BiLSTM\Intent\intents_content.csv') as csv_file:
            data = pd.read_csv(csv_file)
            with open('E:\BHUVANESH\WORKOUT\GP_Mean_covariance\BiLSTM\Intent\generated_questions.csv', mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Intent', 'Generated Question'])
                for index, row in data.iterrows():
                    intent = row["intent"]
                    content = row["content"]
                    generated_questions = generate_questions(tokenizer,model,content)
                    for question in generated_questions:
                        writer.writerow([intent, question])
        print("Questions saved to generated_questions.csv.")

        # Train model on generated questions
        classifier = IntentClassifier('E:\BHUVANESH\WORKOUT\GP_Mean_covariance\BiLSTM\Intent\generated_questions.csv')
        contents, intents = classifier.load_data()
        X, y = classifier.preprocess_data(contents, intents)
        classifier.build_model(128)
        classifier.train_model(X, y)
        classifier.save_model(r'E:\BHUVANESH\WORKOUT\GP_Mean_covariance\BiLSTM\Intent\intent_model.h5')
        while True:
        # Predict intent based on user input
            user_input = input("Enter your question: ")
            predicted_intent = classifier.predict_intent(user_input)
            print("Predicted Intent:", predicted_intent)
    except Exception as e:
        print(f"An error occurred: {e}")
