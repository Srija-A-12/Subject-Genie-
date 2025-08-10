# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import numpy as np
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import LabelEncoder
# import re
# import torch
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# # Download necessary NLP datasets
# nltk.download('stopwords')
# nltk.download('punkt')

# app = Flask(__name__)

# # Load and preprocess dataset
# def load_dataset(csv_file):
#     df = pd.read_csv(csv_file, encoding="utf-8")
#     return df

# def preprocess_text(text):
#     if pd.isna(text):
#         return ""
#     stop_words = set(stopwords.words('english'))
#     words = word_tokenize(text.lower())
#     synonym_map = {
#         "not too hard": "easy", "chill": "easy", "high scoring": "easy", "lenient": "easy",
#         "tough": "hard", "challenging": "hard", "complex": "hard",
#         "practical": "hands-on", "theory-intensive": "theory"
#     }
#     processed_words = [synonym_map.get(word, word) for word in words if word.isalnum() and word not in stop_words]
#     return " ".join(processed_words)

# # Extract relevant info from query
# def extract_info(query):
#     semester_match = re.search(r'\b(?:my\s*)?(\d+)\s*(?:st|nd|rd|th)?\s*semester\b', query, re.IGNORECASE)
#     semester = int(semester_match.group(1)) if semester_match else None

#     category_keywords = {
#         "easy": ["easy", "simple", "high scoring", "lenient", "boost GPA"],
#         "hard": ["hard", "tough", "challenging", "complex"],
#         "theory": ["theory", "theory-based", "theory-intensive"],
#         "practical": ["hands-on", "practical", "applications", "project"],
#         "sums": ["numerical", "sums", "calculations", "problem-solving", "math"],
#         "concept": ["concept", "theoretical", "understanding"],
#         "comparison": ["better", "vs", "should I choose", "is .* harder than", "which is easier"],
#         "feedback": ["popular", "faculty", "student reviews", "grading", "strict grading"],
#         "career": ["career", "job", "useful for", "help me get a job in"]
#     }

#     category = None
#     for key, synonyms in category_keywords.items():
#         if any(re.search(rf'\b{syn}\b', query, re.IGNORECASE) for syn in synonyms):
#             category = key
#             break 

#     domain_match = re.search(r'\b(AI|Artificial Intelligence|Cybersecurity|Data Science|Networking|Cloud Computing|Machine Learning)\b', query, re.IGNORECASE)
#     domain = domain_match.group(1) if domain_match else None

#     return semester, category, domain

# # Train ML model
# def train_model(df, sample_queries):
#     df["Processed_Category"] = df["Category"].apply(preprocess_text)
#     sample_queries["Processed_Query"] = sample_queries["Query"].apply(preprocess_text)

#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(pd.concat([df["Processed_Category"], sample_queries["Processed_Query"]], axis=0))

#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["Elective"].astype(str))

#     X_train, X_test, y_train, y_test = train_test_split(X[:len(df)], y, test_size=0.2, random_state=42)

#     model = LogisticRegression()
#     model.fit(X_train, y_train)

#     return model, vectorizer, label_encoder

# # Load transformer model for generating responses
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
# lm_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

# def generate_response(prompt):
#     inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
#     outputs = lm_model.generate(**inputs, max_length=100)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Recommend electives based on query
# def recommend_elective(query, df, model, vectorizer, label_encoder):
#     semester, category, domain = extract_info(query)

#     if not semester:
#         return "I don't have the required information."

#     filtered_df = df[df["Semester"] == semester]

#     if domain:
#         electives = filtered_df[filtered_df["Domain"].str.contains(domain, case=False, na=False)]
#     elif category:
#         electives = filtered_df[filtered_df["Category"].str.contains(category, case=False, na=False)]
#     else:
#         electives = filtered_df

#     if electives.empty:
#         return "I don't have the required information."

#     recommendations = electives.sample(min(len(electives), 2))[['Elective', 'Domain']].to_dict(orient="records")
#     response = "I recommend: " + ", ".join([f"{rec['Elective']} (Domain: {rec['Domain']})" for rec in recommendations])
#     return response

# # Load datasets
# csv_file = r"C:\Users\adusu\Downloads\electives_dataset_with_domains.csv"  # Ensure the CSV file is present
# sample_queries_file = r"C:\Users\adusu\Downloads\sample_queries.csv"  # New dataset for general queries

# df = load_dataset(csv_file)
# sample_queries = load_dataset(sample_queries_file)

# # Train the model
# model, vectorizer, label_encoder = train_model(df, sample_queries)

# # Flask routes
# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get_response", methods=["POST"])
# def get_response():
#     data = request.get_json()
#     user_query = data.get("query", "")
#     response = recommend_elective(user_query, df, model, vectorizer, label_encoder)
#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(debug=True)




# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder
# import re


# nltk.download('stopwords')
# nltk.download('punkt')

# app = Flask(__name__)

# def load_dataset(csv_file):
#     df = pd.read_csv(csv_file, encoding="utf-8")
#     return df

# def preprocess_text(text):
#     if pd.isna(text):
#         return ""
#     stop_words = set(stopwords.words('english'))
#     words = word_tokenize(text.lower())
#     synonym_map = {
#         "not too hard": "easy", "chill": "easy", "high scoring": "easy", "lenient": "easy",
#         "tough": "hard", "challenging": "hard", "complex": "hard",
#         "practical": "hands-on", "theory-intensive": "theory"
#     }
#     processed_words = [synonym_map.get(word, word) for word in words if word.isalnum() and word not in stop_words]
#     return " ".join(processed_words)

# # Extract relevant info from query
# def extract_info(query):
#     semester_match = re.search(r'\b(?:my\s*)?(\d+)\s*(?:st|nd|rd|th)?\s*semester\b', query, re.IGNORECASE)
#     semester = int(semester_match.group(1)) if semester_match else None

#     category_keywords = {
#         "easy": ["easy", "simple", "high scoring", "lenient", "boost GPA","improve cgpa" ,"cgpa"],
#         "hard": ["hard", "tough", "challenging", "complex"],
#         "theory": ["theory", "theory-based", "theory-intensive"],
#         "practical": ["hands-on", "practical", "applications", "project"],
#         "sums": ["numerical", "sums", "calculations", "problem-solving", "math"],
#         "concept": ["concept", "theoretical", "understanding"],
#         "comparison": ["better", "vs", "should I choose", "is .* harder than", "which is easier"],
#         "feedback": ["popular", "faculty", "student reviews", "grading", "strict grading"],
#         "career": ["career", "job", "useful for", "help me get a job in"]
#     }

#     category = None
#     for key, synonyms in category_keywords.items():
#         if any(re.search(rf'\b{syn}\b', query, re.IGNORECASE) for syn in synonyms):
#             category = key
#             break 

#     domain_match = re.search(r'\b(AI|Artificial Intelligence|Cybersecurity|Data Science|Networking|Cloud Computing|Machine Learning)\b', query, re.IGNORECASE)
#     domain = domain_match.group(1) if domain_match else None

#     return semester, category, domain


# def train_model(df, sample_queries):
#     df["Processed_Category"] = df["Category"].apply(preprocess_text)
#     sample_queries["Processed_Query"] = sample_queries["Query"].apply(preprocess_text)

#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(pd.concat([df["Processed_Category"], sample_queries["Processed_Query"]], axis=0))

#     label_encoder = LabelEncoder()
#     y = label_encoder.fit_transform(df["Elective"].astype(str))

#     X_train, X_test, y_train, y_test = train_test_split(X[:len(df)], y, test_size=0.2, random_state=42)

#     model = LogisticRegression()
#     model.fit(X_train, y_train)

#     return model, vectorizer, label_encoder

# predefined_responses = {
#     "what is cloud computing": "Cloud computing is the delivery of computing services like servers, storage, databases, networking, software, etc., over the internet, or 'the cloud'.",
#     "what is data science": "Data science is a multidisciplinary field that uses scientific methods, processes, algorithms, and systems to extract knowledge and insights from structured and unstructured data.",
#     "what is machine learning": "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data, improve their performance over time, and make predictions.",
#     "what is ai": "Artificial Intelligence (AI) is a field of computer science that aims to create machines that can perform tasks that typically require human intelligence, such as speech recognition, decision-making, and language translation.",
    

#     "hi": "Hello! How can I help you today?",
#     "hii": "Hi there! How can I assist you?",
#     "how are you": "I'm doing great, thanks for asking! How about you?",
#     "need some help": "Sure! What do you need help with?",
#     "need help": "Of course! How can I assist you?",
#     "suggestions": "I'm here to help! Do you want to improve your CGPA, work on a specific domain, or challenge yourself with a difficult elective?",
#     "not sure what elective to take": "Let me help you! Do you want to improve your CGPA, work on a specific domain, or challenge yourself with a difficult elective? Let me know your preference, and I'll suggest some electives."
# }
# def generate_response(query):
#     query = query.strip().lower() 

#     if query in predefined_responses:
#         return predefined_responses[query]

  
#     common_phrases = ["what is", "define", "tell me about", "explain"]
#     for phrase in common_phrases:
#         if query.startswith(phrase):
#             query_without_phrase = query[len(phrase):].strip()
#             if query_without_phrase in predefined_responses:
#                 return predefined_responses[query_without_phrase]
   
#     return "Sorry, I don't have a predefined response for that question. Please try asking something else."


# def recommend_elective(query, df, model, vectorizer, label_encoder):
#     semester, category, domain = extract_info(query)

#     if not semester:
#         return "I don't have the required information."

#     filtered_df = df[df["Semester"] == semester]

#     if domain:
#         electives = filtered_df[filtered_df["Domain"].str.contains(domain, case=False, na=False)]
#     elif category:
#         electives = filtered_df[filtered_df["Category"].str.contains(category, case=False, na=False)]
#     else:
#         electives = filtered_df

#     if electives.empty:
#         return "I don't have the required information."

#     recommendations = electives.sample(min(len(electives), 2))[['Elective', 'Domain']].to_dict(orient="records")
#     response = "I recommend: " + ", ".join([f"{rec['Elective']} (Domain: {rec['Domain']})" for rec in recommendations])
#     return response

# csv_file = r"C:\Users\adusu\Downloads\electives_dataset_with_domains.csv" 
# sample_queries_file = r"C:\Users\adusu\Downloads\sample_queries.csv" 

# df = load_dataset(csv_file)
# sample_queries = load_dataset(sample_queries_file)


# model, vectorizer, label_encoder = train_model(df, sample_queries)


# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get_response", methods=["POST"])
# def get_response():
#     data = request.get_json()
#     user_query = data.get("query", "")
  
#     response = generate_response(user_query)
    
#     if response == "Sorry, I don't have a predefined response for that question. Please try asking something else.":
#         response = recommend_elective(user_query, df, model, vectorizer, label_encoder)
    
#     return jsonify({"response": response})

# if __name__ == "__main__":
#     app.run(debug=True)









# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import re
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk

# nltk.download('stopwords')
# nltk.download('punkt')

# # Initialize Flask app
# app = Flask(__name__)

# # Load the model and tokenizer
# model_name = "microsoft/phi-2"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True
# )

# def load_dataset(csv_file):
#     """Load the dataset and verify required columns exist"""
#     try:
#         df = pd.read_csv(csv_file, encoding="utf-8")
#         required_columns = ['Elective', 'Semester', 'Domain', 'Category', 'Description']
#         missing_columns = [col for col in required_columns if col not in df.columns]
        
#         if missing_columns:
#             raise ValueError(f"Missing required columns in dataset: {missing_columns}")
        
#         return df
        
#     except Exception as e:
#         print(f"Error loading dataset: {str(e)}")
#         return pd.DataFrame()

# def extract_info(query):
#     """Extract semester, category, and domain from user query"""
#     semester_match = re.search(r'\b(?:my\s*)?(\d+)\s*(?:st|nd|rd|th)?\s*semester\b', query, re.IGNORECASE)
#     semester = int(semester_match.group(1)) if semester_match else None
    
#     category_keywords = {
#         "easy": ["easy", "simple", "high scoring", "boost GPA"],
#         "hard": ["hard", "tough", "challenging"],
#         "theory": ["theory", "theory-based"],
#         "practical": ["practical", "hands-on"],
#         "sums": ["numerical", "sums", "problem-solving"],
#     }
    
#     category = None
#     for key, synonyms in category_keywords.items():
#         if any(re.search(rf'\b{syn}\b', query, re.IGNORECASE) for syn in synonyms):
#             category = key
#             break 
    
#     domain_match = re.search(r'\b(AI|Artificial Intelligence|Cybersecurity|Data Science|Cloud Computing)\b', query, re.IGNORECASE)
#     domain = domain_match.group(1) if domain_match else None
    
#     return semester, category, domain

# def recommend_elective(query, df):
#     """Generate elective recommendations based on user query"""
#     semester, category, domain = extract_info(query)
#     if not semester:
#         return "Please specify your semester to get recommendations."
    
#     filtered_df = df[df["Semester"] == semester]
#     if domain:
#         filtered_df = filtered_df[filtered_df["Domain"].str.contains(domain, case=False, na=False)]
#     if category:
#         filtered_df = filtered_df[filtered_df["Category"].str.contains(category, case=False, na=False)]
    
#     if filtered_df.empty:
#         return f"No electives found for semester {semester}. Try different criteria?"
    
#     recommendations = filtered_df.sample(min(len(filtered_df), 2))[['Elective', 'Domain', 'Description']].to_dict(orient="records")
#     response = "Here are some elective suggestions:\n"
#     for i, rec in enumerate(recommendations, 1):
#         response += f"\n{i}. {rec['Elective']} (Domain: {rec['Domain']})\n   Description: {rec['Description']}\n"
#     return response

# def generate_llm_response(prompt):
#     """Generate direct answer response using language model"""
#     try:
#         system_prompt = "You are an academic advisor. Provide direct and informative responses."
#         full_prompt = f"{system_prompt}\nStudent: {prompt}\nAdvisor:"
        
#         inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
#         output = model.generate(
#             **inputs, max_new_tokens=150, temperature=0.7, top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id
#         )
        
#         response = tokenizer.decode(output[0], skip_special_tokens=True)
#         response = response.replace(full_prompt, "").strip()
#         return response.split(".")[0] + "." if "." in response else response + "."
    
#     except Exception as e:
#         print(f"Error in LLM response: {str(e)}")
#         return "I'm having trouble generating a response. Please try again."

# csv_file = r"C:\Users\adusu\Downloads\electives_dataset_with_domains.csv" 
# df = load_dataset(csv_file)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get_response", methods=["POST"])
# def get_response():
#     try:
#         data = request.get_json()
#         user_query = data.get("query", "").strip()
#         if not user_query:
#             return jsonify({"response": "Please enter your query about electives."})
        
#         processed_query = user_query.lower()
#         elective_keywords = ["elective", "course", "subject", "choose", "select", "semester"]
        
#         if any(keyword in processed_query for keyword in elective_keywords):
#             response = recommend_elective(processed_query, df)
#         else:
#             response = generate_llm_response(user_query)
        
#         return jsonify({"response": response})
    
#     except Exception as e:
#         print(f"Error handling request: {str(e)}")
#         return jsonify({"response": "An error occurred. Please try again later."})

# if __name__ == "__main__":
#     app.run(debug=True)

# from flask import Flask, render_template, request, jsonify
# import pandas as pd
# import re
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import nltk

# nltk.download('stopwords')
# nltk.download('punkt')

# # Initialize Flask app
# app = Flask(__name__)

# # Load the model and tokenizer
# model_name = "microsoft/phi-2"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     device_map="auto",
#     trust_remote_code=True
# )

# def load_dataset(csv_file):
#     """Load dataset and check for required columns."""
#     try:
#         df = pd.read_csv(csv_file, encoding="utf-8")
#         required_columns = ['Elective', 'Semester', 'Domain', 'Category']
#         if not all(col in df.columns for col in required_columns):
#             raise ValueError("Dataset is missing required columns.")
#         return df
#     except Exception as e:
#         print(f"Error loading dataset: {e}")
#         return pd.DataFrame()

# def extract_info(query):
#     """Extract semester, category, and domain from user query."""
#     semester_match = re.search(r'\b(\d+)\s*semester\b', query, re.IGNORECASE)
#     semester = int(semester_match.group(1)) if semester_match else None
    
#     category_keywords = {
#         "easy": ["easy", "simple", "boost GPA"],
#         "hard": ["hard", "tough", "challenging"],
#         "theory": ["theory", "theory-based"],
#         "practical": ["hands-on", "practical", "project"],
#     }
#     category = next((key for key, words in category_keywords.items() if any(word in query.lower() for word in words)), None)
    
#     domain_match = re.search(r'\b(AI|Cybersecurity|Data Science|Networking|Cloud Computing)\b', query, re.IGNORECASE)
#     domain = domain_match.group(1) if domain_match else None
    
#     return semester, category, domain

# def recommend_elective(query, df):
#     """Generate elective recommendations."""
#     semester, category, domain = extract_info(query)
#     if not semester:
#         return "Please specify your semester for accurate recommendations."
    
#     electives = df[df["Semester"] == semester]
#     if domain:
#         electives = electives[electives["Domain"].str.contains(domain, case=False, na=False)]
#     elif category:
#         electives = electives[electives["Category"].str.contains(category, case=False, na=False)]
    
#     if electives.empty:
#         return f"No electives found for semester {semester}. Try different criteria?"
    
#     recommendations = electives.sample(min(len(electives), 2))[["Elective", "Domain"]].to_dict(orient="records")
#     response = "Here are some recommended electives:\n"
#     for rec in recommendations:
#         response += f"- {rec['Elective']} ({rec['Domain']})\n"
#     return response

# def generate_llm_response(prompt):
#     """Generate direct response using language model."""
#     try:
#         system_prompt = "You are an academic advisor. Provide clear and direct answers to student queries."
#         full_prompt = f"{system_prompt}\nStudent: {prompt}\nAdvisor:"
        
#         inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
#         output = model.generate(**inputs, max_new_tokens=150, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
#         response = tokenizer.decode(output[0], skip_special_tokens=True).replace(full_prompt, "").strip()
#         response = re.sub(r"\n+", "\n", response)
#         return response.split(".")[0] + "." if "." in response else response + "."
#     except Exception as e:
#         print(f"Error in LLM response: {e}")
#         return "I'm having trouble generating a response. Please try again."

# # Load dataset
# csv_file = r"C:\Users\adusu\Downloads\electives_dataset_with_domains.csv"
# df = load_dataset(csv_file)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/get_response", methods=["POST"])
# def get_response():
#     try:
#         data = request.get_json()
#         user_query = data.get("query", "").strip()
#         if not user_query:
#             return jsonify({"response": "Please enter your query."})
        
#         elective_keywords = ["elective", "course", "subject", "choose", "select", "semester"]
#         response = recommend_elective(user_query, df) if any(word in user_query.lower() for word in elective_keywords) else generate_llm_response(user_query)
#         return jsonify({"response": response})
#     except Exception as e:
#         print(f"Error in request handling: {e}")
#         return jsonify({"response": "An error occurred. Please try again later."})

# if __name__ == "__main__":
#     app.run(debug=True)


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
