from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import semantic_search
from werkzeug.utils import secure_filename
from utils import search_jobs, extract_text, extract_job_title, is_roman_urdu, translate_roman_urdu

app = Flask(__name__)
CORS(app)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'  # Directory where you want to save uploaded files
ALLOWED_EXTENSIONS = {'pdf', 'doc', 'docx'} # Set of allowed file extensions
API_KEY = "api_key = "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""

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """
    Checks if the file extension is allowed.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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
            results_json = results.to_dict(orient='records')
            return jsonify(results_json), 200
        else:
            english_query = translate_roman_urdu(query, API_KEY)
            results = search_jobs(model, index, processed_data, english_query)  # Call the search function
        # Convert the DataFrame to a list of dictionaries before using jsonify
            results_json = results.to_dict(orient='records')
            return jsonify(results_json), 200
    except Exception as e:
        error_message = f"Error during search: {e}"
        print(error_message)
        return jsonify({'error': error_message}), 500
    
@app.route('/api/upload_resume', methods=['POST'])
def upload_resume_api():
    """Handles resume file uploads."""
    if 'resume' not in request.files:
        return jsonify({'error': 'No resume file provided'}), 400
    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            print("Starting resume processing...")

            extracted_text = extract_text(filepath)
            print("Extracted text:", extracted_text[:200])  # preview only

            job_title = extract_job_title(extracted_text, titles)
            print("Predicted job title:", job_title)

            results = search_jobs(model, index, processed_data, job_title)
            print("Search complete. Results found:", len(results))

            results_json = results.to_dict(orient='records')
            return jsonify(results_json), 200
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
