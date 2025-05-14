from flask import Flask, request, jsonify ,redirect, url_for, session
from flask_cors import CORS
import os
import semantic_search
from werkzeug.utils import secure_filename
from utils import calculate_semantic_match_percentage_batched, search_jobs, extract_text, is_roman_urdu, translate_roman_urdu , save_user_info, extract_resume_details,save_resume_data_db, upload_files_to_server
from authlib.integrations.flask_client import OAuth
import pandas as pd
import requests

app = Flask(__name__)
CORS(app, supports_credentials=True)  # Allow all origins for all routes
app.secret_key = os.urandom(24)
oauth = OAuth(app)

google = oauth.register(
    name='google',
    client_id="86197314938-n8vcuc4d842h7ar56qtk0b349l0a3897.apps.googleusercontent.com",
    client_secret="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",
    access_token_url='https://oauth2.googleapis.com/token',
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    userinfo_endpoint='https://openidconnect.googleapis.com/v1/userinfo',
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# Configuration for file uploads
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'} # Set of allowed file extensions
API_KEY = "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"
REMOTE_DIR = '/srv/nfs_share/genai_resumes'
SERVER_IP = "94.136.185.155"
PASSWORD = "1Nu3qW78"
USERNAME = "root"
# Set the UPLOAD_FOLDER (Change the path to your desired directory)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')

def allowed_file(filename):
    """Checks if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/logout')
def logout():
    # Revoke the Google OAuth token (if it exists)
    if 'google_id' in session:
        try:
            token = session.get('oauth_token')
            if token:
                revoke = requests.post(
                    'https://oauth2.googleapis.com/revoke',
                    params={'token': token},
                    headers={'content-type': 'application/x-www-form-urlencoded'}
                )
                print("Google token revoked:", revoke.status_code)
        except Exception as e:
            print("Error revoking Google token:", e)
    
    # Clear all session data
    session.pop('google_id', None)
    session.pop('user_info', None)
    session.clear()
    print("session cleared")

    return jsonify({'success': True}) 


@app.route('/')
def index():
    if 'google_id' in session:
        return f"Logged in as {session['user_info']['name']}! <button onclick=\"window.location.href='/logout'\">Logout</button>"
    return "<a href='/login'>Login with Google</a>"


@app.route('/login')
def login():
    company_url = request.args.get('redirect')
    session['redirect_after_login'] = company_url  # store target
    return google.authorize_redirect(redirect_uri=url_for('auth_callback', _external=True))

@app.route('/auth/callback')
def auth_callback():
    try:
        token = google.authorize_access_token()
        user = google.get('https://openidconnect.googleapis.com/v1/userinfo').json()
        #print("User Info:", user)
        
        print("[DEBUG] saving user to DB")
        save_user_info(user)
        print("[DEBUG] User info saved to DB")
        target_url = session.pop('redirect_after_login', '/')
        # Redirect to home page
        return redirect(f'http://localhost:5173/?redirect={target_url}')
    except Exception as e:
        print(f"[OAuth Error] {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/api/search_jobs', methods=['POST'])
def search_jobs_api():
    """Handles job searches."""
    if model is None or index is None or processed_data is None:
        return jsonify({'error': 'Model or index not loaded.'}), 500

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Missing "query" parameter'}), 400
    query = data['query']

    try:
        if not is_roman_urdu(query):
            results = search_jobs(model, index, processed_data, query)  # Call the search function
        # Convert the DataFrame to a list of dictionaries before using jsonify
            # Replace NaN with None (to convert to JSON null)
            results = results.applymap(lambda x: None if isinstance(x, float) and pd.isna(x) else x)
            df = pd.DataFrame()
            df["combined_text"] = results.apply(
                lambda row: f"{row['title']}. Skills: {row['recruit_skills_ai']}. {row['job_description']}", axis=1)
            matches = calculate_semantic_match_percentage_batched(query, df['combined_text'].tolist())
            match_percentages = [item.get('match_percentage') for item in matches]
            results['match_score'] = match_percentages
            results_sorted = results.sort_values(by='match_score', ascending=False)
            results_json = results_sorted.to_dict(orient='records')
            return jsonify(results_json), 200
        else:
            english_query = translate_roman_urdu(query, API_KEY)
            results = search_jobs(model, index, processed_data, english_query)  # Call the search function
            # Replace NaN with None (to convert to JSON null)
            results = results.applymap(lambda x: None if isinstance(x, float) and pd.isna(x) else x)
            df = pd.DataFrame()
            df["combined_text"] = results.apply(
                lambda row: f"{row['title']}. Skills: {row['recruit_skills_ai']}. {row['job_description']}", axis=1)
            matches = calculate_semantic_match_percentage_batched(english_query, df['combined_text'].tolist())
            match_percentages = [item.get('match_percentage') for item in matches]
            results['match_score'] = match_percentages
            results_sorted = results.sort_values(by='match_score', ascending=False)
            results_json = results_sorted.to_dict(orient='records')
            results_json = results.to_dict(orient='records')
            return jsonify(results_json), 200
    except Exception as e:
        error_message = f"Error during search: {e}"
        print(error_message)
        return jsonify({'error': error_message}), 500
    
@app.route('/api/upload_resume', methods=['POST'])
def upload_resume_api():
    print("upload resume api......")
    triggered_by = request.args.get('triggered_by') or request.headers.get('X-Triggered-By') or request.form.get('triggered_by')
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file provided'}), 400
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        local_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        filepath = f"{REMOTE_DIR}/{filename}"
        try:
            file.save(local_filepath)  # Save the file
            #remote_filepath = f"{REMOTE_DIR}{filename}"
            upload_files_to_server(file, filepath)
            extracted_text = extract_text(local_filepath) # Use the wrapper
            os.remove(local_filepath)
            print("Extracted text:", extracted_text)
            if extracted_text is None: # Check if extraction was successful
                return jsonify({'error': 'Error extracting text from resume'}), 500

            resume_data = extract_resume_details(extracted_text)
            resume_data['text'] = str(extracted_text)
            save_resume_data_db(resume_data)
            if not triggered_by:
                results = search_jobs(model, index, processed_data, resume_data['job_title'].iloc[0])
                df = pd.DataFrame()
                df["combined_text"] = results.apply(
                    lambda row: f"{row['title']}. Skills: {row['recruit_skills_ai']}. {row['job_description']}", axis=1)
                matches = calculate_semantic_match_percentage_batched(resume_data['text'], df["combined_text"].tolist())
                match_percentages = [item.get('match_percentage') for item in matches]
                results['match_score'] = match_percentages
                results_sorted = results.sort_values(by='match_score', ascending=False)
                results_json = results_sorted.to_dict(orient='records')
                results_json = results.to_dict(orient='records')
                return jsonify(results_json), 200
            else:
                return jsonify({'message': 'Resume successfully uploaded'}), 200  # Simple success message
        except Exception as e:
            import traceback
            traceback.print_exc()
            error_message = f"Error processing resume: {e}"
            return jsonify({'error': error_message}), 500
    else:
        return jsonify({'error': 'Invalid file type. Allowed types are pdf, doc, docx'}), 400

if __name__ == '__main__':
    model, index, processed_data, titles = semantic_search.semantic()
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
