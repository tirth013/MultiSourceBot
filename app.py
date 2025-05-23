from flask import Flask, render_template, request, jsonify
from main import research
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

# Research endpoint
@app.route('/research', methods=['POST'])
def do_research():
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        result = research(query)
        return jsonify({'result': result})
    except Exception as e:
        # Production safe error handling - don't expose details
        return jsonify({'error': 'An error occurred processing your request'}), 500

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Page not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Production configuration
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
