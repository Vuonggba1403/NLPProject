from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle
import numpy as np
from load_datasets import load_skills_list, load_education_keywords
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

# Xác định đường dẫn đến thư mục models
MODELS_DIR = BASE_DIR / 'models'

# Load models
with open(MODELS_DIR / 'best_model_categorization.pkl', 'rb') as f:
    best_model_categorization = pickle.load(f)

with open(MODELS_DIR / 'tfidf_vectorizer_categorization.pkl', 'rb') as f:
    tfidf_vectorizer_categorization = pickle.load(f)

with open(MODELS_DIR / 'gb_classifier_job_recommendation.pkl', 'rb') as f:
    gb_classifier_job_recommendation = pickle.load(f)

with open(MODELS_DIR / 'tfidf_vectorizer_job_recommendation.pkl', 'rb') as f:
    tfidf_vectorizer_job_recommendation = pickle.load(f)
# Load datasets from Excel
skills_list = load_skills_list()
education_keywords = load_education_keywords()

# Clean resume
def cleanResume(txt):
    # Use a simpler approach with a character class
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~]', ' ', cleanText)  # Fixed punctuation removal
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

# Prediction functions
def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    return best_model_categorization.predict(resume_tfidf)[0]

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    return gb_classifier_job_recommendation.predict(resume_tfidf)[0]

def pdf_to_text(file):
    reader = PdfReader(file.stream)
    text = ''
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

# GPA extractor
def extract_gpa_from_resume(text):
    gpa_patterns = [
        r'GPA[:\s]*([0-4]\.\d{1,2}\s*/\s*4\.0)',  # GPA: 3.5/4.0
        r'GPA[:\s]*([0-4]\.\d{1,2})',              # GPA: 3.5
        r'([0-4]\.\d{1,2}\s*/\s*4\.0)',             # 3.5/4.0
        r'([0-4]\.\d{1,2})'           # 3.5/4.0
    ]
    for pattern in gpa_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None

# Contact info extractors
def extract_contact_number_from_resume(text):
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

def extract_email_from_resume(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

# Skills extractor
def extract_skills_from_resume(text):
    skills = []
    for skill in skills_list:
        if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
            skills.append(skill)
    return skills

# Education extractor
def extract_education_from_resume(text):
    education = []
    for keyword in education_keywords:
        if re.search(rf"(?i)\b{re.escape(keyword)}\b", text):
            education.append(keyword)
    return education

# Name extractor
def extract_name_from_resume(text):
    pattern = r"\b([A-Z][a-z]{1,}\s){1,2}[A-Z][a-z]{1,}\b"
    match = re.search(pattern, text)
    return match.group() if match else None

# Routes
@app.route('/')
def resume():
    return render_template("resume.html")

@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename

        if filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file.")

        data = {
            'predicted_category': predict_category(text),
            'recommended_job': job_recommendation(text),
            'phone': extract_contact_number_from_resume(text),
            'email': extract_email_from_resume(text),
            'extracted_skills': extract_skills_from_resume(text),
            'extracted_education': extract_education_from_resume(text),
            'name': extract_name_from_resume(text),
            'gpa': extract_gpa_from_resume(text),
        }

        return render_template('resume.html', **data)
    else:
        return render_template("resume.html", message="No resume file uploaded.")

if __name__ == '__main__':
    app.run(debug=True)