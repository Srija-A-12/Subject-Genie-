from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('stopwords')
nltk.download('punkt')

app = Flask(__name__)

# Load model & tokenizer
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load dataset
def load_dataset(csv_file):
    try:
        df = pd.read_csv(csv_file, encoding="utf-8")
        required_columns = ['Elective', 'Semester', 'Domain', 'Category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return pd.DataFrame()

def extract_info(query):
    semester_match = re.search(r'(\d+)\s*(?:st|nd|rd|th)?\s*semester', query, re.IGNORECASE)
    semester = int(semester_match.group(1)) if semester_match else None

    category_keywords = {
        "easy": ["easy", "high scoring", "improve cgpa"],
        "hard": ["hard", "tough", "challenging"],
        "theory": ["theory", "theory-based"],
        "practical": ["practical", "hands-on", "project"],
        "sums": ["numerical", "sums", "calculations"],
        "industry": ["industry", "job", "career", "demand"]
    }
    category = next((key for key, words in category_keywords.items() if any(re.search(fr'\b{word}\b', query, re.IGNORECASE) for word in words)), None)
    
    domain_match = re.search(r'\b(AI|Machine Learning|Cybersecurity|Data Science|Networking|Cloud Computing)\b', query, re.IGNORECASE)
    domain = domain_match.group(1) if domain_match else None
    
    return semester, category, domain

def recommend_elective(query, df):
    semester, category, domain = extract_info(query)
    if not semester:
        return "Please specify your semester for accurate recommendations."
    
    filtered_df = df[df["Semester"] == semester]
    if domain:
        filtered_df = filtered_df[filtered_df["Domain"].str.contains(domain, case=False, na=False)]
    if category:
        filtered_df = filtered_df[filtered_df["Category"].str.contains(category, case=False, na=False)]
    
    if filtered_df.empty:
        return f"No electives found for semester {semester}. Try different criteria?"
    
    recommendations = filtered_df.sample(min(len(filtered_df), 2))[['Elective', 'Domain']].to_dict(orient="records")
    response = "Here are some elective suggestions:\n"
    response += "\n".join([f"{i+1}. {rec['Elective']} (Domain: {rec['Domain']})" for i, rec in enumerate(recommendations)])
    return response

def generate_llm_response(prompt):
    try:
        full_prompt = f"You are an academic advisor.\n\nStudent: {prompt}\nAdvisor:"
        inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        output = model.generate(**inputs, max_new_tokens=150, temperature=0.7, top_p=0.9, repetition_penalty=1.1, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(output[0], skip_special_tokens=True).replace(full_prompt, "").strip()
        return response.split(".")[0] + "."
    except Exception as e:
        print(f"LLM error: {str(e)}")
        return "I'm having trouble generating a response. Please try again."

csv_file = "C:/Users/adusu/Downloads/electives_dataset_with_domains.csv"
df = load_dataset(csv_file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        data = request.get_json()
        user_query = data.get("query", "").strip()
        if not user_query:
            return jsonify({"response": "Please enter your query."})
        
        if any(keyword in user_query.lower() for keyword in ["elective", "semester", "subject"]):
            response = recommend_elective(user_query, df)
        else:
            response = generate_llm_response(user_query)
        
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error handling request: {str(e)}")
        return jsonify({"response": "An error occurred. Please try again."})

if __name__ == "__main__":
    app.run(debug=True)
