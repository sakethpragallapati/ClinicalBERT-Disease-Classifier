import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import warnings

# Suppress some transformers warnings for cleaner output
warnings.filterwarnings("ignore")

def load_model(model_dir="./clinical_bert_disease_classifier", mapping_file="label_mapping.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model to {device}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        
        with open(mapping_file, "r") as f:
            label_mapping = json.load(f)
            
        return model, tokenizer, device, label_mapping
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Make sure you have run train.py first to generate the model in {model_dir}")
        return None, None, None, None

def predict_disease(text, model, tokenizer, device, label_mapping, top_k=3):
    model.eval()
    
    # 8. Convert symptoms -> clinical sentence (handled by the model directly via tokenization)
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    # 9. Disease probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Get top k probabilities and indices
    top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(label_mapping)))
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        disease_name = label_mapping[str(idx.item())]
        results.append({
            "disease": disease_name,
            "probability": prob.item() * 100
        })
        
    return results

def main():
    model, tokenizer, device, label_mapping = load_model()
    if model is None:
        return
        
    print("\n" + "="*50)
    print("ClinicalBERT Disease Classifier")
    print("="*50)
    print("Describe your symptoms to predict the possible disease.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    
    while True:
        query = input("\nSymptoms: ")
        if query.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
            
        if not query.strip():
            continue
            
        predictions = predict_disease(query, model, tokenizer, device, label_mapping)
        
        top_prob = predictions[0]['probability']
        
        print("\n--- Predictions ---")
        if top_prob < 75.0:
            print("⚠️  [LOW CONFIDENCE] The model is not highly certain, but here are the best guesses:")
            
        for i, pred in enumerate(predictions, 1):
            print(f"{i}. {pred['disease']:<20} | Probability: {pred['probability']:.2f}%")

if __name__ == "__main__":
    main()
