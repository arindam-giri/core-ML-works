import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from transformers.trainer import Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from datasets import Dataset
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

# This script fine-tunes a DistilBERT model for binary text classification on a Kaggle dataset.
# 1. Dataset Loading and Preprocessing
# Load the Kaggle train.csv dataset
df = pd.read_csv('train.csv')
logging.info("Dataset loaded successfully.")
# Clean the comment_text column
# Convert to string and handle null values
df['comment_text'] = df['comment_text'].astype(str).fillna('')  # Convert to string, replace NaN with empty string
logging.info("Comment text cleaned and NaN values handled.")
# Map toxicity labels to binary "safe" (0) and "unsafe" (1)
def create_binary_label(row):
    is_toxic = any([
        row['toxic'],
        row['severe_toxic'],
        row['obscene'],
        row['threat'],
        row['insult'],
        row['identity_hate']
    ])
    return 1 if is_toxic else 0
logging.info("Binary labels created for toxicity classification.")
df['labels'] = df.apply(create_binary_label, axis=1)

# Select relevant columns
df = df[['comment_text', 'labels']]
logging.info("Relevant columns selected: comment_text and labels.")
# Split into train (80%) and test (20%) sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
logging.info("Dataset split into train and test sets.")
# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_df.reset_index(drop=True))
test_dataset = Dataset.from_pandas(test_df.reset_index(drop=True))
logging.info("Converted DataFrames to Hugging Face Datasets.")
logging.info(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")   
# 2. Tokenization
# Use DistilBERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
logging.info("DistilBERT tokenizer loaded successfully.")
def tokenize_function(examples):
    # Tokenize text with truncation and padding
    return tokenizer(
        examples['comment_text'],
        padding="max_length",
        truncation=True,
        max_length=128
    )

# Tokenize datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
logging.info("Train dataset tokenized successfully.")
tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
logging.info("Test dataset tokenized successfully.")
# Set format for PyTorch, excluding comment_text to avoid type issues
tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
logging.info("Datasets formatted for PyTorch.")
# 3. Model Selection and Fine-Tuning
# Load DistilBERT for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# Define training arguments optimized for Colab's free-tier GPU
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=100,
    report_to="none"
)

# Define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
logging.info("Training arguments and metrics function defined.")
# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    compute_metrics=compute_metrics,
)
logging.info("Trainer initialized successfully.")
logging.info("Starting model training...")
# Fine-tune the model
trainer.train()
logging.info("Model training completed successfully.")  
# 4. Evaluation
# Evaluate on test set
eval_results = trainer.evaluate()
print("Evaluation Results:")
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Precision: {eval_results['eval_precision']:.4f}")
print(f"Recall: {eval_results['eval_recall']:.4f}")
print(f"F1-Score: {eval_results['eval_f1']:.4f}")

# 5. Inference Examples
def classify_prompt(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=128)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
    return "unsafe" if prediction == 1 else "safe"

# Example prompts
example_prompts = [
    "Thank you for your help, I appreciate it!",
    "You are an idiot and should be fired.",
    "Can you assist with my account issue?",
    "I hate this company and everyone in it."
]

print("\nInference Examples:")
for prompt in example_prompts:
    label = classify_prompt(prompt)
    print(f"Prompt: {prompt}")
    print(f"Classification: {label}\n")

# 6. Save the model and tokenizer
model.save_pretrained("./fine_tuned_distilbert")
tokenizer.save_pretrained("./fine_tuned_distilbert")