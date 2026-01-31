import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
import os

def run_batch(input_path, output_path):
    model_path = "./model_output"
    if not os.path.exists(model_path):
        print("Error: Train the model first!")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    df = pd.read_csv(input_path)
    results = []
    
    print(f"Processing {len(df)} rows...")
    for text in df['text']:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            pred = torch.argmax(outputs.logits, dim=-1).item()
        results.append("positive" if pred == 1 else "negative")
    
    df['predicted_sentiment'] = results
    df.to_csv(output_path, index=False)
    print(f"Success! Batch results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run_batch(args.input, args.output)
