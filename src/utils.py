# -*- coding: utf-8 -*-
import pandas as pd
import re
from sentence_transformers import SentenceTransformer, models, losses
from torch.utils.data import DataLoader
import torch
from sentence_transformers import InputExample
import numpy as np
import shutil
import os
import zipfile
import faiss
import fitz
from docx import Document
import requests
import mysql.connector
from mysql.connector import Error
from loguru import logger
from bs4 import BeautifulSoup


def db_connection_and_data():
    try:
        connection = mysql.connector.connect(
          host='13.233.151.156',
          user= 'root',
          password= '4T_V0Llw4c&mxrhz',
          database= 'workisy_worksuite',
          port= '3306'
        )

        if connection and connection.is_connected(): #Check if connection is valid
            query = "SELECT * FROM recruit_jobs"
            df = pd.read_sql(query, connection) #Use pandas to read
            return df
        else:
            print("Error: No database connection.")
            return None

    except Error as e:
        print(f"Error while connecting to MySQL or reading data: {e}")
        return None

    finally:
        if connection and connection.is_connected():
            connection.close()
            print("MySQL connection is closed")


def clean_text(text):
    text = str(text)

    # Step 0: Strip HTML tags with BeautifulSoup
    text = BeautifulSoup(text, "html.parser").get_text(separator=' ')

    # Step 1: Add spaces between camelCase transitions (optional)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

    # Step 2: Lowercase
    text = text.lower()

    # Step 3: Replace special characters with space
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Step 4: Remove tag names that often remain as text (like 'p', 'strong', 'span')
    text = re.sub(r'\b(p|strong|span|div|br|ul|li|ol|em|b|i)\b', ' ', text)

    # Step 5: Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def data_preprocessing(data):
    print("Data preprocessed started")
    for col in ['title', 'recruit_skills_ai', 'job_description']:
        data[col] = data[col].apply(clean_text)
    data["combined_text"] = data.apply(
        lambda row: f"{row['title']}. Skills: {row['recruit_skills_ai']}. {row['job_description']}", axis=1)
    all_titles = data["title"].dropna().str.strip().str.lower().unique().tolist()
    print("Data preprocessed")
    return data, all_titles


def upzip_saved_model():
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    zip_file_path = os.path.join(script_dir, 'fine_tuned_sbert_job_search1.zip')
    zip_file_dir = os.path.dirname(zip_file_path)
    extract_path = None
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(zip_file_dir)
    print("Model unzipped")
    extract_path = os.path.join(zip_file_dir, 'fine_tuned_sbert_job_search1')
    return extract_path

def load_saved_model(extract_path):
    model = SentenceTransformer(extract_path)
    print("Model Loaded")
    return model

def fine_tuning(data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = SentenceTransformer('all-MiniLM-L6-v2', use_auth_token=False, device=device)
    examples = [InputExample(texts=[text, text]) for text in data["combined_text"]]
    train_dataloader = DataLoader(examples, shuffle=True, batch_size=16)
    train_loss = losses.MultipleNegativesRankingLoss(model)
    os.environ["WANDB_DISABLED"] = "true"
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10)
    model.save("./fine_tuned_sbert_job_search1")
    shutil.make_archive("fine_tuned_sbert_job_search1", 'zip', "./fine_tuned_sbert_job_search1") #add proper values in quotes
    print("Model saved!")

    print("Fine-tuning complete!")


def faiss_index(model, data):
    print("Creating embeddings")
    job_embeddings = model.encode(data["combined_text"].tolist(), normalize_embeddings=True, convert_to_tensor=True)
    print("Embeddings created successfully")
    job_embeddings_cpu = job_embeddings.cpu().numpy()
    dimension = job_embeddings_cpu.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using L2 (Euclidean) distance for search
    index.add(job_embeddings_cpu)
    print(f"FAISS index with {index.ntotal} vectors created.")
    faiss.write_index(index, 'job_search_index1.index')
    print("FAISS index saved as 'job_search_index.index'")
    # Load FAISS index
    index = faiss.read_index('job_search_index1.index')
    print("FAISS index loaded successfully!")

    return index

def search_jobs(model,index,data,query, top_k=10):
    # Encode the query to get its embedding (assuming query_embedding is on GPU)
    print("Searching jobs")
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Move query_embedding to CPU and convert it to numpy
    query_embedding_cpu = query_embedding.cpu().numpy()

    # Perform the search in the FAISS index
    D, I = index.search(np.array([query_embedding_cpu]), k=top_k)

    # Fetch the matching job titles
    results = data.iloc[I[0]]

    return results[['title', 'recruit_skills_ai', 'job_description', 'recruit_locations_ai','recruit_salary_ai','updated_at','recruit_work_exp_ai']]

def is_roman_urdu(text):
    roman_urdu_keywords = {'main', 'hoon', 'mera', 'ka', 'ki', 'tum', 'kya', 'nahi', 'hai', 'job', 'mujhe', 'cahiye'}
    if not re.fullmatch(r"[A-Za-z0-9\s\.,!?'\-]+", text):
        return False  # Contains non-Latin script or symbols
    words = set(text.lower().split())
    matched = words & roman_urdu_keywords
    return len(matched) >= 2  # At least 2 matches

def translate_roman_urdu(text, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Instruct the model to ONLY return the English translation, nothing else
    prompt = (
        f'You are a translator. Translate this Roman Urdu sentence to English: "{text}". '
        'Only return the translated sentence without special characters. Do not explain anything.'
    )

    data = {
        "model": "mistralai/mistral-7b-instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()['choices'][0]['message']['content']
        return result.strip().strip('"')  # Clean up extra quotes or whitespace
    else:
        return f"Error: {response.status_code} - {response.text}"

def extract_text_from_pdf(file_path):
    text = ""
    pdf = fitz.open(file_path)
    for page in pdf:
        text += page.get_text()
    return text

def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

# Detect file type and extract text
def extract_text(file_path):
    if file_path.endswith('.pdf'):
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF or DOCX file.")

def extract_job_title(text, all_titles):

    # Create a regex pattern from the job titles list
    pattern = r'(' + '|'.join(map(re.escape, all_titles)) + ')'

    # Use re.findall to search for matching job titles in the text
    possible_titles = re.findall(pattern, text, re.IGNORECASE)

    if possible_titles:
        return possible_titles[0]  # Return the first match
    else:
        return "Could not find job title"
