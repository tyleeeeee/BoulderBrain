from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

'''
Here we define endpoints for each functionality.
Then these will call the respective functions in the service layer.
'''


@app.route('/')
def home():
    return "Hello from Flask!"

@app.route('/api/data')
def data():
    return jsonify({"message": "Ty hacked the back end!"})

if __name__ == '__main__':
    app.run(debug=True)
