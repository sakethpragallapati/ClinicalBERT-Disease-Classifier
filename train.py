import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # 1. Load dataset
    print("Loading dataset...")
    df = pd.read_csv("datasets/Combined_Dataset.csv")
    
    # 2. Encode labels
    label_encoder = LabelEncoder()
    df['encoded_label'] = label_encoder.fit_transform(df['label'])
    
    # Save label mapping for later use
    label_mapping = {int(index): label for index, label in enumerate(label_encoder.classes_)}
    with open("label_mapping.json", "w") as f:
        json.dump(label_mapping, f)
        
    num_labels = len(label_mapping)
    print(f"Total disease classes: {num_labels}")
    
    # 3. Train / validation / test split (80% Train, 10% Val, 10% Test)
    # Stratify ensures each disease has equal representation in splits
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['encoded_label'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['encoded_label'])
    
    # Convert to HuggingFace Datasets
    # We rename 'encoded_label' to 'label' as expected by HuggingFace Trainer
    train_dataset = Dataset.from_pandas(train_df[['text', 'encoded_label']].rename(columns={'encoded_label': 'label'}))
    val_dataset = Dataset.from_pandas(val_df[['text', 'encoded_label']].rename(columns={'encoded_label': 'label'}))
    test_dataset = Dataset.from_pandas(test_df[['text', 'encoded_label']].rename(columns={'encoded_label': 'label'}))
    
    # 4. ClinicalBERT tokenizer
    MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
        
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Remove unnecessary columns to avoid tensor conversion errors
    cols_to_remove = ['text']
    if '__index_level_0__' in train_dataset.column_names:
        cols_to_remove.append('__index_level_0__')
        
    train_dataset = train_dataset.remove_columns(cols_to_remove)
    val_dataset = val_dataset.remove_columns(cols_to_remove)
    test_dataset = test_dataset.remove_columns(cols_to_remove)
    
    train_dataset.set_format("torch")
    val_dataset.set_format("torch")
    test_dataset.set_format("torch")
    
    # 5. Pretrained Bio_ClinicalBERT + Classification Head
    print("Loading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for fine-tuning: {device}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
    ).to(device)
    
    # 6. Fine-tuning on GPU (with fixes for overfitting)
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=15, # Increased max epochs, but EarlyStopping will halt it early
        learning_rate=2e-5, # Lower learning rate to prevent jumping into overfit regions
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_ratio=0.1, # Use ratio instead of static steps
        weight_decay=0.05, # Increased weight decay for stronger regularization
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", # Optimize for lowest loss instead of highest accuracy
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Stops if val loss doesn't improve for 2 epochs
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    print("Evaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print(f"Test Results: {test_results}")
    
    # 7. Save model
    save_directory = "./clinical_bert_disease_classifier"
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

if __name__ == "__main__":
    main()
