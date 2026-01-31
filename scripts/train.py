import os
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # Requirement 6: Calculate precision, recall, f1, and accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1)
    }

def train_model():
    # Load processed data (Requirement 4)
    train_df = pd.read_csv('data/processed/train.csv')
    test_df = pd.read_csv('data/processed/test.csv')

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenization logic from Colab
    def tokenize_func(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=256)

    # Prepare datasets for Trainer
    from datasets import Dataset
    train_ds = Dataset.from_pandas(train_df).map(lambda x: tokenize_func(x['text']), batched=True)
    test_ds = Dataset.from_pandas(test_df).map(lambda x: tokenize_func(x['text']), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Use the same hyperparameters as your Colab run (Requirement 7)
    training_args = TrainingArguments(
        output_dir='./results/checkpoints',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics
    )

    trainer.train()
    
    # Requirement 6 & 7: Generate metrics and summary
    metrics = trainer.evaluate()
    
    os.makedirs('results', exist_ok=True)
    with open('results/metrics.json', 'w') as f:
        json.dump({k: v for k, v in metrics.items() if k in ['accuracy', 'precision', 'recall', 'f1_score']}, f)

    run_summary = {
        "hyperparameters": {
            "model_name": model_name,
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 3
        },
        "final_metrics": {
            "accuracy": metrics['eval_accuracy'],
            "f1_score": metrics['eval_f1_score']
        }
    }
    with open('results/run_summary.json', 'w') as f:
        json.dump(run_summary, f, indent=4)

    # Requirement 5: Save artifacts
    os.makedirs('model_output', exist_ok=True)
    model.save_pretrained('model_output')
    tokenizer.save_pretrained('model_output')

if __name__ == "__main__":
    train_model()