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
import re
class ini_int_train:
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
        # Check and create the directory if it doesn't exist
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        # Save the model using the checkpoint API for easier restore
        checkpoint_path = os.path.join(self.checkpoint_dir, 'ckpt')
        save_path = self.checkpoint.save(checkpoint_path)
        print(f"Model checkpoint saved at {save_path}")

        # Save tokenizer's word index as JSON
        tokenizer_path = os.path.join(self.checkpoint_dir, 'tokenizer.json')
        with open(tokenizer_path, 'w', encoding='utf-8') as f:
            json.dump(self.tokenizer.word_index, f, ensure_ascii=False)

        # Save labels as JSON
        labels_path = os.path.join(self.checkpoint_dir, 'labels.json')
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(self.labels, f, ensure_ascii=False)
        print("Tokenizer and labels saved.")


    def train_model(self, X, y, epochs=100, batch_size=32):
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
        
    def split_and_save_csv(input_file_path):
        try:
            # Read the CSV file into a DataFrame
            data = pd.read_csv(input_file_path, encoding='latin1')

            # Convert the list of tuples into a DataFrame
            df = pd.DataFrame(data, columns=['id', 'parent_id', 'intent'])

            # Establish a mapping from each id to its parent id and intent
            parent_mapping = df.set_index('id')['parent_id'].to_dict()
            intent_mapping = df.set_index('id')['intent'].to_dict()

            # Initialize the output structure
            output = []

            for id, intent in intent_mapping.items():
                current_id = id
                path = [intent]

                # Trace back the path to the initial intent
                while parent_mapping[current_id] != 0:
                    current_id = parent_mapping[current_id]
                    path.insert(0, intent_mapping[current_id])

                # Prepare the row for output data
                row = {
                    'intent': path[0],
                    'refined_intent': intent
                }

                # Add sub-intents dynamically based on the path length
                for i, p in enumerate(path[1:], start=1):
                    row[f'sub_intent{i}'] = p

                output.append(row)

            # Convert output to DataFrame
            output_df = pd.DataFrame(output)

            # Ensure all possible sub-intent columns are present, fill missing values with None
            max_sub_intents = output_df.columns.str.extract(r'sub_intent(\d+)').dropna().astype(int).max()
            for i in range(1, max_sub_intents[0]+1):
                if f'sub_intent{i}' not in output_df.columns:
                    output_df[f'sub_intent{i}'] = None

            # Reorder columns to have sub_intents in order and at the end
            sub_intent_cols = [col for col in output_df if col.startswith('sub_intent')]
            other_cols = [col for col in output_df if col not in sub_intent_cols]
            output_df = output_df[other_cols + sorted(sub_intent_cols, key=lambda x: int(x.replace('sub_intent','')))]
            output_df.fillna('none', inplace=True)

            # adding text column
            text = pd.DataFrame(data, columns=['text'])
            output_df['content'] = text
            output_df.dropna(inplace=True)
            initial_intent_data = output_df[['intent', 'content']]

            # Read the initial intent DataFrame
            df = initial_intent_data

            # List to hold new split rows
            new_rows = []

            # Iterate through each row in the DataFrame
            for index, row in df.iterrows():
                # Normalize the content by replacing newlines and other whitespace characters with a single space
                normalized_content = re.sub(r'\s+', ' ', row['content'])

                # Split the 'content' using regex to find a period followed by one or more whitespace characters
                split_parts = re.split(r'\.\s+', normalized_content)

                for part in split_parts:
                    # Append new row with the same 'intent' and the normalized and split 'content'
                    new_rows.append({'intent': row['intent'], 'content': part})

            # Create a new DataFrame with the new split rows
            split_df = pd.DataFrame(new_rows)

            # Specify the output file path
            output_file_path = input_file_path.replace('.csv', '_processed.csv')

            # Save the new split DataFrame to a CSV file
            split_df.to_csv(output_file_path, index=False)

            # Return the path to the processed output CSV file
            return output_file_path
        
        except Exception as e:
            print("An error occurred:", e)
            return None

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
            checkpoint_dir = os.path.join('BiLSTM','intent', 'model', 'nn_model')
            
            
            tokenizert5 = T5Tokenizer.from_pretrained("google/flan-t5-base")
            modelt5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")
            
            preprocess_data = 'E:\BHUVANESH\WORKOUT\GP_Mean_covariance\BiLSTM\Intent\data\web_scrap_data.csv'
            processed_data = ini_int_train.split_and_save_csv(preprocess_data)
            with open(os.path.join('BiLSTM\intent\data', 'web_scrap_data_processed.csv')) as processed_data:
                data = pd.read_csv(processed_data)
                # Generate questions from content data and save to CSV
                with open(os.path.join('BiLSTM\intent\data', 'generated_questions.csv'), mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerow(['Intent', 'Generated Question'])
                    for index, row in data.iterrows():
                        intent = row["intent"]
                        content = row["content"]
                        generated_questions = ini_int_train.generate_questions(tokenizert5, modelt5, content)
                        for question in generated_questions:
                            writer.writerow([intent, question]) 
            print("Questions saved to generated_questions.csv.")
            
            questions_csv_path = os.path.join('BiLSTM', 'intent', 'data', 'generated_questions.csv')

            classifier = ini_int_train(questions_csv_path, checkpoint_dir)
            # Call this function at the appropriate time after setting up self.labels
            contents, intents = classifier.load_data()
            if contents and intents:  # Ensure data is loaded
                X, y = classifier.preprocess_data(contents, intents)
            
            
                classifier.build_model()
                classifier.train_model(X, y)
                prompt = 'Classifying intial intent using LSTM neural network model train and stored successfully'
                desc = 'Success'
                status = '1'
                result = status, desc, prompt
                return result
            
        except Exception as e:
            desc = f"An error occurred: {e}"
            print(f"An error occurred: {e}")
            prompt = ''
            desc = desc
            status = '0'
            result = status, desc, prompt
            return result

ini_int_train.main()

