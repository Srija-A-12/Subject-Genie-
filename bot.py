import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from flask import Flask, render_template, request, jsonify

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

def load_dataset(csv_file):
    df = pd.read_csv(csv_file, encoding="utf-8")
    return df

def preprocess_text(text):
    if pd.isna(text):
        return ""
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    synonym_map = {
        "not too hard": "easy", "chill": "easy", "high scoring": "easy", "lenient": "easy",
        "tough": "hard", "challenging": "hard", "complex": "hard",
        "practical": "hands-on", "theory-intensive": "theory"
    }
    processed_words = [synonym_map.get(word, word) for word in words if word.isalnum() and word not in stop_words]
    return " ".join(processed_words)

def extract_info(query):
    semester_match = re.search(r'\b(?:my\s*)?(\d+)\s*(?:st|nd|rd|th)?\s*semester\b', query, re.IGNORECASE)
    semester = int(semester_match.group(1)) if semester_match else None
    category_keywords = {
        "easy": ["easy", "simple", "high scoring", "lenient", "boost GPA"],
        "hard": ["hard", "tough", "challenging", "complex"],
        "theory": ["theory", "theory-based", "theory-intensive"],
        "practical": ["hands-on", "practical", "applications", "project"],
        "sums": ["numerical", "sums", "calculations", "problem-solving", "math"],
        "concept": ["concept", "theoretical", "understanding"],
    }
    category = None
    for key, synonyms in category_keywords.items():
        if any(re.search(rf'\b{syn}\b', query, re.IGNORECASE) for syn in synonyms):
            category = key
            break 
    domain_match = re.search(r'\b(AI|Artificial Intelligence|Cybersecurity|Data Science|Networking|Cloud Computing|Machine Learning)\b', query, re.IGNORECASE)
    domain = domain_match.group(1) if domain_match else None
    return semester, category, domain

def train_model(df):
    df["Processed_Category"] = df["Category"].apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["Processed_Category"])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Elective"].astype(str))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, vectorizer, label_encoder

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
lm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = lm_model.generate(**inputs, max_length=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def recommend_elective(query, df, model, vectorizer, label_encoder):
    semester, category, domain = extract_info(query)
    if not semester:
        return "I don't have the required information."
    filtered_df = df[df["Semester"] == semester]
    if domain:
        electives = filtered_df[filtered_df["Domain"].str.contains(domain, case=False, na=False)]
    elif category:
        electives = filtered_df[filtered_df["Category"].str.contains(category, case=False, na=False)]
    else:
        electives = filtered_df
    if electives.empty:
        return "I don't have the required information."
    recommendations = electives.sample(min(len(electives), 2))[['Elective', 'Domain']].to_dict(orient="records")
    response = "I recommend: " + ", ".join([f"{rec['Elective']} (Domain: {rec['Domain']})" for rec in recommendations])
    return response

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    response = recommend_elective(user_input, df, model, vectorizer, label_encoder)
    return jsonify({"response": response})

if __name__ == "__main__":
    csv_file = "electives_dataset_with_domains.csv"
    df = load_dataset(csv_file)
    model, vectorizer, label_encoder = train_model(df)
    app.run(debug=True)
