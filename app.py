from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle
import numpy as np
import os
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

with open(MODELS_DIR / 'xgboost_model_job_recommendation.pkl', 'rb') as f:
    xgb_model_data = pickle.load(f)

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
    
    # Check if best_model_categorization is XGBoost (stored as dictionary)
    if isinstance(best_model_categorization, dict) and best_model_categorization.get("type") == "xgboost":
        # For XGBoost, we need to handle the label encoding
        xgb_model = best_model_categorization["model"]
        label_encoder = best_model_categorization["label_encoder"]
        
        # Get the encoded prediction
        encoded_pred = xgb_model.predict(resume_tfidf)[0]
        
        # Convert back to original category name
        return label_encoder.inverse_transform([encoded_pred])[0]
    else:
        # Standard models (Logistic Regression, Naive Bayes, Random Forest)
        return best_model_categorization.predict(resume_tfidf)[0]

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    
    # Get XGBoost model and label mapping from the loaded data
    xgb_model = xgb_model_data["model"]
    label_mapping = xgb_model_data["label_mapping"]
    
    # Make prediction with XGBoost (returns encoded value)
    encoded_pred = xgb_model.predict(resume_tfidf)[0]
    
    # Convert encoded prediction back to original job category
    return label_mapping[encoded_pred]

def pdf_to_text(file):
    # Create a copy of the file stream for reading
    file_copy = file.stream
    reader = PdfReader(file_copy)
    text = ''
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    # Reset the file pointer for later saving
    file.seek(0)
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

def save_resume_file(file, category):
    # Create the category directory if it doesn't exist
    save_dir = os.path.join(BASE_DIR, 'Saves', category)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the file with its original name
    file_path = os.path.join(save_dir, file.filename)
    
    # Ensure the file pointer is at the beginning
    file.seek(0)
    file.save(file_path)
    
    # Return the relative path for display purposes
    return os.path.join('Saves', category, file.filename)

@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename

        if not filename:
            return render_template('resume.html', message="No file selected.")
        
        if filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif filename.endswith('.txt'):
            # Read the file content
            text = file.read().decode('utf-8')
            # Reset the file pointer for later saving
            file.seek(0)
        else:
            return render_template('resume.html', message="Invalid file format. Please upload a PDF or TXT file.")

        # Get the predicted category
        predicted_category = predict_category(text)
        
        # Save the file to the appropriate category folder
        saved_path = save_resume_file(file, predicted_category)

        data = {
            'predicted_category': predicted_category,
            'recommended_job': job_recommendation(text),
            'phone': extract_contact_number_from_resume(text),
            'email': extract_email_from_resume(text),
            'extracted_skills': extract_skills_from_resume(text),
            'extracted_education': extract_education_from_resume(text),
            'name': extract_name_from_resume(text),
            'gpa': extract_gpa_from_resume(text),
            'saved_path': saved_path,
        }

        return render_template('resume.html', **data)
    else:
        return render_template("resume.html", message="No resume file uploaded.")

if __name__ == '__main__':
    app.run(debug=True)