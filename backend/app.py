from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello from Flask!"

@app.route('/api/data')
def data():
    return jsonify({"message": "Ben hacked the back end!"})

if __name__ == '__main__':
    app.run(debug=True)
