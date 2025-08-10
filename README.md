# 🎓 Elective Recommendation System

A Flask-based academic elective recommendation tool that helps students choose electives based on their semester, domain, and category preferences. This system combines rule-based NLP filtering with a Microsoft Phi-2 large language model (LLM) to provide personalized academic advice. 🤖✨

---

## 🚀 Features

- 🔍 Parses user queries to extract semester, category (e.g., easy, hard, theory), and domain (e.g., AI, Data Science).  
- 📊 Filters electives from a custom-curated dataset based on extracted criteria.  
- 🧠 Uses Microsoft Phi-2 transformer model for natural language response generation.  
- 💻 Simple and interactive Flask web interface.

---

## 👩‍💼 Your Role

- 🏆 Led the project, coordinating development and paper submission.  
- 💡 Conceptualized the idea and designed the recommendation logic.  
- 📚 Collected and curated the electives dataset with relevant metadata.  
- 🔍 Researched and selected the LLM model to use.  
- 🛠️ Contributed to regex-based query parsing and model integration.

---

## ⚙️ Setup Instructions

1. Clone the repository:  
   ```
   git clone https://github.com/Srija-A-12/Subject-Genie-.git
   cd Subject-Genie-
2.Install dependencies:
 ```
pip install flask pandas torch transformers nltk

 ```
3.Run the Flask app:
 ```
python app.py
 ```

4.Open your browser at http://127.0.0.1:5000 to use the system. 🌐

# 🗂️ Dataset
The electives_dataset_with_domains.csv file contains the elective data with columns:
* Elective
* Semester
* Domain
* Category

📝 Notes
* The model uses GPU for better performance. ⚡
* NLTK stopwords and tokenizers are required. 📥
* Feel free to customize the dataset or add more query parsing features. 🔧



# MIT License

Copyright (c) 2025 Srija A

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

