# load_datasets.py
import pandas as pd
from pathlib import Path

# Xác định thư mục gốc của dự án (nơi đặt load_datasets.py)
BASE_DIR = Path(__file__).resolve().parent

def load_skills_list():
    file_path = BASE_DIR / 'datasets' / 'skills_dataset.xlsx'
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file kỹ năng tại: {file_path}")
    
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    return df['Skills'].dropna().astype(str).str.strip().tolist()

def load_education_keywords():
    file_path = BASE_DIR / 'datasets' / 'Education.xlsx'
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file giáo dục tại: {file_path}")
    
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    return df['Education'].dropna().astype(str).str.strip().tolist()
