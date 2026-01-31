import os
import pandas as pd
from datasets import load_dataset
import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)  # Remove HTML line breaks
    text = re.sub(r'[^a-z0-9\s]', '', text) # Remove special characters
    return text.strip()

def preprocess_data():
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb")
    
    # Convert to DataFrames
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    # BALANCING LOGIC: Essential for 90% accuracy
    # We take 2500 positive and 2500 negative to ensure no bias
    pos_train = train_df[train_df['label'] == 1].sample(2500, random_state=42)
    neg_train = train_df[train_df['label'] == 0].sample(2500, random_state=42)
    balanced_train = pd.concat([pos_train, neg_train]).sample(frac=1, random_state=42)

    # Do the same for test set
    pos_test = test_df[test_df['label'] == 1].sample(500, random_state=42)
    neg_test = test_df[test_df['label'] == 0].sample(500, random_state=42)
    balanced_test = pd.concat([pos_test, neg_test]).sample(frac=1, random_state=42)

    print("Cleaning text data...")
    balanced_train['text'] = balanced_train['text'].apply(clean_text)
    balanced_test['text'] = balanced_test['text'].apply(clean_text)

    # Requirement 3: Create data/processed directory and save CSVs
    os.makedirs('data/processed', exist_ok=True)
    
    balanced_train[['text', 'label']].to_csv('data/processed/train.csv', index=False)
    balanced_test[['text', 'label']].to_csv('data/processed/test.csv', index=False)
    
    print(f"Success! Saved {len(balanced_train)} training and {len(balanced_test)} test samples.")

if __name__ == "__main__":
    preprocess_data()