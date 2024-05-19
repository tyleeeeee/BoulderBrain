from .hold import Hold
from .wall import Wall

import os
# from ultralytics import YOLO

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
        Hold(dummy_wall, [50, 100], "blue1", False, [50, 105]),  # Lower left
        Hold(dummy_wall, [150, 120], "green1", False, [150, 125]),  # Mid left
        Hold(dummy_wall, [250, 150], "yellow1", False, [250, 155]),  # Center wall
        Hold(dummy_wall, [350, 180], "green2", False, [350, 185]),  # Mid right
        Hold(dummy_wall, [450, 210], "red1", False, [450, 215]),  # Lower right
        Hold(dummy_wall, [100, 220], "blue2", False, [100, 225]),  # Mid upper left
        Hold(dummy_wall, [200, 250], "red2", True, [200, 255]),  # Upper center
        Hold(dummy_wall, [300, 280], "yellow2", True, [300, 285]),  # Mid upper right
        Hold(dummy_wall, [400, 310], "red3", True, [400, 315]),  # Upper right
        Hold(dummy_wall, [50, 340], "blue3", True, [50, 345]),  # Top left
        Hold(dummy_wall, [150, 370], "green3", True, [150, 375]),  # Top mid left
        Hold(dummy_wall, [250, 400], "yellow3", True, [250, 405]),  # Top center
    ]

def generate_dense_holds(wall):
    holds = []
    max_reach = 70  # climber's max reach for simplicity
    vertical_spacing = max_reach * 0.3
    horizontal_spacing = max_reach * 0.3

    # Calculate how many holds can fit based on spacing and wall dimensions.
    num_vertical = int(wall.height / vertical_spacing)
    num_horizontal = int(wall.width / horizontal_spacing)

    # Generate holds in a grid-like pattern.
    for v in range(num_vertical):
        for h in range(num_horizontal):
            x = h * horizontal_spacing + (vertical_spacing / 2 if v % 2 == 1 else 0)
            y = v * vertical_spacing
            if x <= wall.width and y <= wall.height:
                holds.append(Hold(wall, [x, y], "green", False, [x, y + 5]))

    # Add some final holds at the top of the wall for good measure.
    for h in range(num_horizontal):
        x = h * horizontal_spacing + (vertical_spacing / 2 if v % 2 == 1 else 0)
        y = wall.height
        if x <= wall.width: holds.append(Hold(wall, [x, y], "green", False, [x, y + 5]))

    return holds