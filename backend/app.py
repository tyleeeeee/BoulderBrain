from flask import Flask, jsonify, request
from flask_cors import CORS
from services.image_processing_service import get_holds_from_image
from services.route_generation_service import generateRoutes

from services.items.climber import Climber
from services.items.wall import Wall

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Hello from Flask!"

@app.route('/api/data')
def data():
    return jsonify({"message": "Ty hacked the back end!"})

@app.route('/api/generate_routes', methods=['GET'])
def api_generate_routes():
    try:
        # Initialize Wall and Climber
        wall = Wall(id=1, height=10, width=5)
        climber = Climber(wall, height=180, upper_arm_length=40, forearm_length=30,
                          upper_leg_length=45, lower_leg_length=40, torso_height=80,
                          torso_width=50)

        # Set up a new wall with holds
        wall.holds = get_holds_from_image()

        # generate routes
        routes = generateRoutes(wall, climber)

        if not routes:
            raise ValueError("No routes could be generated with the current setup.")
        return jsonify({"routes": routes}) # best for frontend

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
