        # output = {
        #     'Req': {
        #         'Type': 'GCND',
        #         'neuron_list': [{'neuron_id': str(id)} for id in classifier]  # Format as requested
        #     }
        # }
        # return output

import pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration
import warnings
warnings.filterwarnings("ignore")
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Embedding
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

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
        self.checkpoint = tf.train.Checkpoint()

    def build_model(self):
        if self.labels is None:
            raise ValueError("Labels are not set. Cannot build model without label information.")
        
        self.model = Sequential([
            Embedding(input_dim=len(self.tokenizer.word_index) + 1, output_dim=self.output_dim),
            Bidirectional(LSTM(64)),
            Dense(len(self.labels), activation='softmax')
        ])
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.checkpoint.model = self.model

    def save_model_and_tokenizer(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        # Save model in native Keras format
        model_path = os.path.join(self.checkpoint_dir, 'intent_model.keras')
        self.model.save(model_path)
        
        # Save tokenizer's word index as JSON
        tokenizer_path = os.path.join(self.checkpoint_dir, 'tokenizer.json')
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            json.dump(self.tokenizer.word_index, f)
        
        labels_path = os.path.join(self.checkpoint_dir, 'labels.json')
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f)
        print("Model and tokenizer checkpoint saved.")

    def train_model(self, X, y, epochs=1000, batch_size=32):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        self.save_model_and_tokenizer()

    def load_data(self):
        try:
            df = pd.read_csv(self.csv_path)
            contents = df['Generated Question'].tolist()
            intents = df['Intent'].tolist()
            return contents, intents
        except Exception as e:
            print(f"Failed to read the CSV file: {e}")
            return [], []

    def preprocess_data(self,contents, intents):
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
        

    def generate_questions(tokenizert5, modelt5, content):
        try:
            input_text = f"Based on the following information, generate ten questions whose answers can be directly found in the text: \"{content}\" Ensure each question corresponds to specific details or facts mentioned."
            input_ids = tokenizert5(input_text, return_tensors="pt").input_ids
            outputs = modelt5.generate(
                input_ids,
                max_length=200,
                num_return_sequences=10,
                num_beams=10,
                temperature=0.9,
                no_repeat_ngram_size=1
            )
            generated_questions = [tokenizert5.decode(output, skip_special_tokens=True) for output in outputs]
            return generated_questions
        except Exception as e:
            print(f"Error generating questions: {e}")
            return []

    def main():
        try:
            
            checkpoint_dir = os.path.join('apps', 'static', 'data', 'model', 'checkpoints')
            
            # if os.path. exists(checkpoint_dir):
            #     os.remove(checkpoint_dir)
            
            tokenizert5 = T5Tokenizer.from_pretrained("google/flan-t5-base")
            modelt5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
            # Generate questions from content data and save to CSV
            with open(os.path.join('apps\static\data', 'intents_content.csv')) as csv_file:
                data = pd.read_csv(csv_file)
                with open(os.path.join('apps\static\data', 'generated_questions.csv'), mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Intent', 'Generated Question'])
                    for index, row in data.iterrows():
                        intent = row["intent"]
                        content = row["content"]
                        generated_questions = IntentClassifier.generate_questions(tokenizert5, modelt5, content)
                        for question in generated_questions:
                            writer.writerow([intent, question]) 
            print("Questions saved to generated_questions.csv.")

            # base_path = r'/apps/static/model'
            # checkpoint_dir = os.path.join(base_path, 'model')
            
            questions_csv_path = os.path.join('apps', 'static', 'data', 'generated_questions.csv')

            classifier = IntentClassifier(questions_csv_path, checkpoint_dir)
            # Call this function at the appropriate time after setting up self.labels
            contents, intents = classifier.load_data()
            if contents and intents:  # Ensure data is loaded
                X, y = classifier.preprocess_data(contents, intents)
            
            # if tf.train.latest_checkpoint(classifier.checkpoint_dir) is None:
                classifier.build_model()
                classifier.train_model(X, y)
            # else:   
            #     print("test1")

        except Exception as e:
            print(f"An error occurred: {e}")
        

IntentClassifier.main()
