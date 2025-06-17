import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, render_template, jsonify
import torch.nn.functional as F

# Initialize Flask app
app = Flask(__name__)

# Set device (Use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model details
MODEL_PATH = "./saved_model"
MODEL_NAME = "xlm-roberta-base"  # Ensure it matches the trained model

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
if os.path.exists(MODEL_PATH):
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
else:
    raise FileNotFoundError("Saved Tamil Fake News model not found. Train and save the model first.")

model.to(device)
model.eval()

# Tamil-specific label mapping
label_names = ["True | உண்மை", "False | பொய்"]

# Function to preprocess input text
def preprocess_text(text):
    return text.strip().lower().replace(".", "").replace(",", "")

# Prediction function
def verify_news(news_text):
    news_text = preprocess_text(news_text)
    inputs = tokenizer(news_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1)
    confidence, prediction = torch.max(probs, dim=1)
    return label_names[prediction.item()], confidence.item()

# Flask routes
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    news_text = request.form.get("news")
    if not news_text:
        return jsonify({"error": "Please enter a news statement."})
    
    prediction, confidence = verify_news(news_text)
    return jsonify({"prediction": prediction, "confidence": f"{confidence:.2f}"})

if __name__ == '__main__':
    app.run(debug=True)
