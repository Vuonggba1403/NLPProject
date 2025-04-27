# load_datasets.py
import pandas as pd

def load_skills_list():
    df = pd.read_excel('FinalProject/datasets/skills_dataset.xlsx', sheet_name='Sheet1')
    return df['Skills'].dropna().astype(str).str.strip().tolist()

def load_education_keywords():
    df = pd.read_excel('FinalProject/datasets/Education.xlsx', sheet_name='Sheet1')
    return df['Education'].dropna().astype(str).str.strip().tolist()
