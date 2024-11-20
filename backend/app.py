# Important!
# Before running, enter the virtual environment by using .\.venv\Scripts\activate.
# (Make sure you are at \Desktop\BoulderBrain first)

from flask import Flask, jsonify
from flask_cors import CORS
# from services.image_processing_service import get_holds_from_image
from services.image_processing_service import get_holds_main
# from services.image_processing_service import generate_dense_holds, get_holds_from_image
from services.route_generation_service import generateRoutes, process_final_routes, filter_routes_by_hold_overlap
from services.reachable_foot_area import calc_knee_angle, calc_hold_angle, calc_hip_angle, calc_max_hip_angle
from services.output_processing import output_route
from services.climber import Climber
from services.wall import Wall

import random
import copy

#gggg
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
        # image_path = 'services/files/example_wall.jpg' 
        image_path = 'services/files/example_wall.jpg'
        print("Loading wall and climber details...")

        wall = Wall(id=0, height=350, width=700, image_path=image_path) #made it larger on purpose
        climber = Climber(wall, height=160, upper_arm_length=25, forearm_length=20,
                          upper_leg_length=40, lower_leg_length=40, torso_height=40,
                          torso_width=30)

        # Set up a new wall with holds
        holds_path = f'services/result{wall.id}/holds'
        holds_img_path = f'services/result{wall.id}/holds_img'
        files_path = f'services/result{wall.id}'
        wall.holds,holds_map = get_holds_main(wall, image_path, holds_path, files_path)

        direction = input("Enter 'v' to generate vertical routes, and 'h' to generate horizontal routes: ")


        # If the climber is going horizontally, they can reach further, so let's stretch their limbs out.
        horizontalBoostFactor = 2

        if direction == 'h':
            climber.upper_arm_length *= horizontalBoostFactor
            climber.forearm_length *= horizontalBoostFactor
            climber.lower_leg_length *= horizontalBoostFactor
            climber.upper_leg_length *= horizontalBoostFactor

        print("Generating routes...")
        routes = generateRoutes(wall, climber, direction)
        holds_dict, routes_description_dict, route_difficulties_dict = process_final_routes(routes)
    
        # overlap_threshold = 30  # TODO: adjust where? Frontend? Try again when tree grows longer
        overlap_threshold = int(input("Insert your maximum overlap threshold: "))
        # print("OT:", overlap_threshold )
        print("Getting valid routes...")
        valid_routes, valid_difficulties = filter_routes_by_hold_overlap(holds_dict, overlap_threshold, wall, route_difficulties_dict, direction)
        # print("Valid Routes:", valid_routes)    
        # Convert set to list to make it serializable
        valid_routes_list = list(valid_routes)
        print(f"Total of {len(valid_routes)} valid routes.")

        output_route(wall.holds, holds_map, valid_routes, valid_difficulties, wall.image_path, files_path)
        
        # return jsonify(valid_routes)
        return jsonify({'Valid Routes': valid_routes})


    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

