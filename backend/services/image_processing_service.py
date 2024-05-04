
from items.climber import Climber
from items.hold import Hold
from items.position import Position
from items.route import Route
from items.wall import Wall
import os
from ultralytics import YOLO

# Print the current working directory to confirm it
print("Current Working Directory:", os.getcwd())


# Path to the image file (assuming you've confirmed the current working directory is the project root)
image_path = 'climbingWallTestImage.jpg'

# Load and run YOLO model
model = YOLO('yolov9c-seg.pt')
result = model(image_path, show=True, save=True)


#def getBodyFromImage(image):
  # 1. Call Mediapipe to extract pose.

#def getValidHolds(image):
# 1. Call SAM to extract segments
# 2. Label segments as "hold" or "not hold".
#    Option 2a: Train a classifier model (finding data could be hard)
#    Option 2b: Use a formula to determine if a segment is a hold
# 3. For each hold, create a hold object with its location, shape, and color.
# 4. Return the set of holds.