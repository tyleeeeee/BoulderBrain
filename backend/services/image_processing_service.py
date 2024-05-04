from .hold import Hold
from .wall import Wall

import os
from ultralytics import YOLO

# # Print the current working directory to confirm it
# print("Current Working Directory:", os.getcwd())
#
#
# # Path to the image file (assuming you've confirmed the current working directory is the project root)
# image_path = 'climbingWallTestImage.jpg'
#
# # Load and run YOLO model
# model = YOLO('yolov9c-seg.pt')
# result = model(image_path, show=True, save=True)


#def getBodyFromImage(image):
  # 1. Call Mediapipe to extract pose.

def get_holds_from_image():
    #TODO: this is a placeholder! Change code later!
    # 1. Call SAM to extract segments
    # 2. Label segments as "hold" or "not hold".
    #    Option 2a: Train a classifier model (finding data could be hard)
    #    Option 2b: Use a formula to determine if a segment is a hold
    # 3. For each hold, create a hold object with its location, shape, and color.
    # 4. Return the set of holds.

    dummy_wall = Wall(id=1, height=400, width=500)
    return [
        Hold(dummy_wall, [1, 1], "blue1", False, [1, 1.5]),  # Near the bottom-left
        Hold(dummy_wall, [2, 1], "green1", False, [2, 1.5]),  # Near the bottom-center
        Hold(dummy_wall, [3, 1], "red1", False, [3, 1.5]),  # Near the bottom-right
        Hold(dummy_wall, [1, 3], "blue2", False, [1, 3.5]),  # Mid-left
        Hold(dummy_wall, [2, 3], "yellow1", False, [2, 3.5]),  # Mid-center
        Hold(dummy_wall, [3, 3], "green2", False, [3, 3.5]),  # Mid-right
        Hold(dummy_wall, [1, 5], "red2", True, [1, 5.5]),  # Top-left
        Hold(dummy_wall, [2, 5], "blue3", True, [2, 5.5]),  # Top-center
        Hold(dummy_wall, [3, 5], "yellow2", True, [3, 5.5])  # Top-right
    ]