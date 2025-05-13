# Resume Screening AI System

## Overview
This project is a Resume Screening AI system that automatically categorizes resumes, recommends suitable jobs, and extracts important information from uploaded resume files. The system uses machine learning models trained on resume data to perform these tasks efficiently.

## Features
- **Resume Categorization**: Categorizes resumes into various professional fields (e.g., IT, Healthcare, HR)
- **Job Recommendation**: Suggests suitable job roles based on resume content
- **Information Extraction**:
  - Personal details (name, email, phone)
  - Education qualifications and GPA
  - Skills and competencies
- **File Management**: Automatically saves uploaded resumes in folders organized by categories
- **Multiple File Format Support**: Accepts PDF and TXT resume files

## Project Structure
```
├── app.py                                 # Main Flask application
├── load_datasets.py                       # Functions to load skill & education datasets
├── Resume Job Recommendation System.ipynb # Notebook for job recommendation model
├── Resume_Catogorization_prediction.ipynb # Notebook for resume categorization model
├── datasets/                              # Training & reference data
│   ├── Education.xlsx                     # Education keywords database
│   └── skills_dataset.xlsx                # Skills database
├── models/                                # Trained machine learning models
│   ├── best_model_categorization.pkl      # Resume categorization model
│   ├── tfidf_vectorizer_categorization.pkl # TF-IDF vectorizer for categorization
│   ├── tfidf_vectorizer_job_recommendation.pkl # TF-IDF vectorizer for job recommendation
│   └── xgboost_model_job_recommendation.pkl # XGBoost model for job recommendation
├── Saves/                                 # Where uploaded resumes are saved
│   └── [CATEGORY]/                        # Subfolders for each category (created automatically)
└── templates/                             # HTML templates
    └── resume.html                        # Main user interface template
```

## Technologies Used
- **Backend**: Python with Flask web framework
- **Machine Learning**: 
  - XGBoost for job recommendation
  - Various classification models for resume categorization
  - TF-IDF vectorization for text processing
- **Text Processing**: 
  - Regular expressions for information extraction
  - PyPDF2 for PDF parsing
- **Frontend**: HTML/CSS with responsive design

## How It Works

### Resume Processing Pipeline
1. User uploads a resume file (PDF or TXT)
2. System extracts text content from the file
3. Text is cleaned and preprocessed
4. Machine learning models analyze the text to:
   - Categorize the resume
   - Recommend suitable jobs
5. Information extraction algorithms identify:
   - Personal details
   - Educational background
   - Professional skills
6. The resume file is saved in the appropriate category folder
7. Results are displayed to the user

### Machine Learning Models
- **Categorization Model**: Classifies resumes into professional categories
- **Job Recommendation Model**: Suggests relevant job roles using XGBoost

## Setup Instructions

### Prerequisites
- Python 3.x
- Required Python packages:
  ```
  Flask
  PyPDF2
  numpy
  pandas
  scikit-learn
  xgboost
  ```

### Installation
1. Clone the repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Ensure all dataset files and models are in their respective directories

### Running the Application
1. Navigate to the project directory
2. Run the Flask application:
   ```
   python app.py
   ```
3. Open a web browser and go to: `http://127.0.0.1:5000/`

## Usage
1. Access the web interface
2. Upload a resume file (PDF or TXT format)
3. Click "Analyze Resume"
4. View the analysis results:
   - Resume category
   - Recommended job
   - Extracted personal information
   - Skills and education details
5. The system automatically saves the resume in the appropriate category folder

## File Saving Structure
Uploaded resumes are automatically saved in the following structure:
```
Saves/
├── IT/                 # Category folder for IT resumes
│   ├── resume1.pdf
│   └── resume2.txt
├── HEALTHCARE/         # Category folder for Healthcare resumes
│   └── resume3.pdf
└── HR/                 # Category folder for HR resumes
    └── resume4.pdf
```

## Future Improvements
- Support for more resume file formats (DOCX, HTML)
- Enhanced information extraction capabilities
- Interactive dashboard for managing multiple resumes
- Integration with job posting platforms
- Export functionality for resume analysis results

## License
[Specify License Information]

## Contributors
[List Project Contributors]
