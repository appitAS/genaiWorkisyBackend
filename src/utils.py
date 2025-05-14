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
import traceback
import paramiko
from groq import Groq
from docx import Document
import requests
import mysql.connector
from mysql.connector import Error
from loguru import logger
from bs4 import BeautifulSoup
from datetime import datetime

GROQ_API_KEY = "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"
REMOTE_DIR = '/srv/nfs_share/genai_resumes'
SERVER_IP = "94.136.185.155"
PASSWORD = "1Nu3qW78"
USERNAME = "root"
client = Groq(api_key=GROQ_API_KEY)

def db_connection_and_data():
    try:
        print("Attempting to connect to MySQL...")
        connection = mysql.connector.connect(
            host='13.233.151.156',
            user='root',
            password='4T_V0Llw4c&mxrhz',
            database='workisy_worksuite',
            port='3306'
        )
        print(connection, "connection")

        if connection and connection.is_connected():
            print("Connection successful.")
            query = "SELECT * FROM recruit_jobs"
            df = pd.read_sql(query, connection)
            print(f"Total number of rows: {len(df)}")
            if not df.empty:
                print("\nColumns in the table:")
                for column in df.columns:
                    print(f"- {column}")
            return df
        else:
            print("Error: No database connection.")
            return None

    except Error as e:
        print(f"Error while connecting to MySQL or reading data: {e}")
        return None

    finally:
        if 'connection' in locals() and connection and connection.is_connected():
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
    zip_file_path = os.path.join(script_dir, 'fine_tuned_sbert_job_search.zip')
    zip_file_dir = os.path.dirname(zip_file_path)
    extract_path = None
    
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(zip_file_dir)
    print("Model unzipped")
    extract_path = os.path.join(zip_file_dir, 'fine_tuned_sbert_job_search')
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
    return results[['title', 'recruit_skills_ai', 'job_description', 'recruit_locations_ai','recruit_salary_ai','updated_at','recruit_work_exp_ai','Link']]

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

def extract_resume_details(resume_text):
    """
    Extracts name, job title, years of experience, and skills from resume text using the Groq API.

    Args:
        resume_text: The full text content of the resume.

    Returns:
        A dictionary containing the extracted information (name, job_title, years_of_experience, skills).
    """

    prompt = f"""
    From the following resume text, please extract the following information and return it as a JSON object:

    {{
      "name": "The full name of the person in this resume",
      "job_title": "The current or most recent job title held by this person",
      "years_of_experience": "The total years of professional experience presented in this resume (calculate by summing the duration of all listed roles. If only a year is provided, assume it's a full year. If 'Present' is used as an end date, consider the current year as {datetime.now().year})",
      "skills": "A list of all technical skills mentioned in the resume"
      "location": "The permanent location of the person in the resume"
      "phone": "the phone number of the person in the resume"
      "email": "the email address of the person in the resume"
      "years of exp": "Only professional work experience of ther person in this resume"
    }}

    **Resume Text:**
    {resume_text}
    """

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Or another suitable Groq model
            messages=[
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )

        if response.choices and response.choices[0].message.content:
            extracted_data = response.choices[0].message.content
            # Convert JSON string to dictionary
            extracted_data_dict = eval(extracted_data)
            # Convert to DataFrame
            ed = pd.DataFrame([extracted_data_dict])
            return ed
        else:
            return {"error": "Could not extract information from the resume."}

    except Exception as e:
        return {"error": f"An error occurred: {e}"}

def save_user_info(user):
    print("[DEBUG] Connecting to MySQL Database...")
    db = mysql.connector.connect(
        host='13.233.151.156',
        user= 'root',
        password= '4T_V0Llw4c&mxrhz',
        database= 'workisy_worksuite',
        port= '3306'
    )
    try:
        if db and db.is_connected(): #Check if connection is valid
            print("[DEBUG] Connecting to MySQL Database...")
            cursor = db.cursor()

            query = """
            INSERT INTO workisy_google_auth_data (sub, name, given_name, family_name, picture, email, email_verified, locale)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                name = VALUES(name),
                given_name = VALUES(given_name),
                family_name = VALUES(family_name),
                picture = VALUES(picture),
                email = VALUES(email),
                email_verified = VALUES(email_verified),
                locale = VALUES(locale);
            """
            
            values = (
                user.get("sub"),
                user.get("name"),
                user.get("given_name"),
                user.get("family_name"),
                user.get("picture"),
                user.get("email"),
                user.get("email_verified"),
                user.get("locale")
            )
            
            print("[DEBUG] Executing SQL Query:", query)
            print("[DEBUG] With Values:", values)

            cursor.execute(query, values)
            db.commit()
            cursor.close()
            print("User information saved successfully.")
        else: 
            print("Error: No database connection.")
            return None
    except mysql.connector.Error as e:
        print(f"[Database Error] {e}")
    except Exception as e:
        print(f"[Unexpected Error] {e}")

def save_resume_data_db(resume_text):
    db = mysql.connector.connect(
        host='13.233.151.156',
        user= 'root',
        password= '4T_V0Llw4c&mxrhz',
        database= 'workisy_worksuite',
        port= '3306'
    )
    try:
        if db and db.is_connected(): #Check if connection is valid
            print("[DEBUG] Connecting to MySQL Database...")
            cursor = db.cursor()
            resume_text['skills'] = resume_text['skills'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
            for index, row in resume_text.iterrows():
                cursor.execute("""
                    INSERT INTO workisy_extracted_resume_data (name, job_title, email, phone, location, text, skills, years_of_exp)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                        name = VALUES(name),
                        job_title = VALUES(job_title),
                        phone = VALUES(phone),
                        location = VALUES(location),
                        text = VALUES(text),
                        skills = VALUES(skills),
                        years_of_exp = VALUES(years_of_exp)
                """, (row['name'], row['job_title'], row['email'], row['phone'], row['location'], row['text'], row['skills'], row['years_of_experience']))

            db.commit()
            cursor.close()
            print("data saved successfully.")
        else: 
            print("Error: No database connection.")
            return None
    except mysql.connector.Error as e:
        print(f"[Database Error] {e}")
    except Exception as e:
        print(f"[Unexpected Error] {e}")

def upload_files_to_server(file, filepath):
    try:
        # Establish SSH connection using Paramiko
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(SERVER_IP, username=USERNAME, password=PASSWORD)
        print("Connected to the remote server.")

        # Use SFTP to upload the file directly
        sftp = ssh.open_sftp()
        sftp.putfo(file.stream, filepath)  # Directly upload file from the stream
        sftp.close()
        print("File uploaded to remote server at:", filepath)

        return {'message': f'File uploaded successfully to {filepath}'}, 200

    except Exception as e:
        traceback.print_exc()
        return {'error': f"Unexpected error: {str(e)}"}, 500

    finally:
        ssh.close()
        print("SSH connection closed.")

def calculate_semantic_match_percentage_batched(query_or_resume_text: str, search_results: list[str], model_name: str = "llama-3.1-8b-instant") -> list[dict]:
    """
    Calculates the semantic match percentage of a query or resume text against a list of search results
    using a Groq language model for direct similarity scoring.

    Args:
        query_or_resume_text: The query or extracted resume text to compare.
        search_results: A list of strings representing the search results.
        model_name: The name of the Groq language model to use for scoring
                    (default: "llama-3.1-8b-instant").

    Returns:
        A list of dictionaries, where each dictionary contains a search result and its match percentage (0-100).
    """
    scored_results = []
    for result in search_results:
        prompt = f"""You are an expert at evaluating the semantic similarity between a query and a result. Your evaluation should focus on the degree to which the result accurately and completely addresses the core meaning and intent of the query.

            Provide a similarity score on a scale of 0 to 100, where:
            - 100 means the result is a perfect semantic match to the query, conveying the exact same meaning and addressing all aspects of it comprehensively.
            - 75-99 indicates a high degree of semantic similarity, where the result captures the main meaning of the query but might have minor differences in nuance or completeness.
            - 50-74 suggests a moderate degree of similarity, where there's a noticeable overlap in meaning, but significant aspects of the query might be missing or misinterpreted.
            - 25-49 indicates low semantic similarity, where the result has some superficial connection to the query but fails to capture its core meaning.

            Consider the following aspects when determining the similarity score:
            - **Core meaning and intent:** Does the result address the central idea and purpose behind the query?
            - **Completeness:** Does the result provide all the necessary information or address all parts of the query?
            - **Nuance and context:** Does the result understand and respect the subtle aspects and context of the query?
            - **Factual accuracy (if applicable):** If the query pertains to factual information, is the result factually correct and consistent with the query's implied requirements?

            provide ONLY the numerical similarity score for the following:

        Text 1: {query_or_resume_text}
        Text 2: {result}
        Similarity Score (0-100): """

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=5  # Limit tokens for a quick score
            )
            content = response.choices[0].message.content
            match = re.search(r'(\d+(\.\d+)?)', content)
            if match:
                score = float(match.group(1))
                scored_results.append({"result": result, "match_percentage": score})
            else:
                print(f"Could not extract score from response for result: '{result}'. Response: {content}")
                scored_results.append({"result": result, "match_percentage": 0.0})
        except Exception as e:
            print(f"Error during similarity scoring for result: '{result}': {e}")
            scored_results.append({"result": result, "match_percentage": 0.0})

    return scored_results
