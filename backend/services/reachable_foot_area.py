import math
from scipy.spatial import distance
import numpy as np
from .climber import Climber
from .wall import Wall
from .position import Position

def calc_knee_angle(hold_location, hip_location, climber):
  # For a hold location and a hip location (2D-coordinate), this function uses trigonometry
  # to determine what the angle of the knee is, assuming the foot is on the hold.

  # According to the cosine law, c^2 = a^2 + b^2 -2ab * cos(theta), 
  # where theta is the angle between a and b. 
  # Then theta = arccos((c^2 - a^2 - b^2) / -2ab)

  # The distance from the hip to the hold is the length of one side of a triangle (c), 
  # and the upper and lower leg lengths are the other two side lengths (a and b).

  a = climber.upper_leg_length
  b = climber.lower_leg_length
  c = distance.euclidean(hold_location, hip_location)

  numerator = a**2 + b**2 - c**2
  denominator = 2 * a * b
  if denominator == 0: return math.pi / 2  # Handle division by zero case

  return math.degrees(math.acos(numerator / denominator))

def calc_hold_angle(hold_location, hip_location, climber):
  a = climber.upper_leg_length
  b = climber.lower_leg_length
  c = distance.euclidean(hold_location, hip_location)

  numerator = a**2 + c**2 - b**2
  denominator = 2 * a * c
  if denominator == 0: return math.pi / 2  # Handle division by zero case

  return math.degrees(math.acos(numerator / denominator))
  

def calc_hip_angle(hold_location, hip_location, shoulder_location, climber):

  # Calculate hold angle, and convert to radains for math.cos and math.sin.
  holdAngle = math.radians(calc_hold_angle(hold_location, hip_location, climber))

  # Calculate hip-to-hold vector.
  hipToHold = [hold_location[0] - hip_location[0], hold_location[1] - hip_location[1]]

  # Rotate vector by the hold angle, so it points at the knee, and make it an numpy array.
  hipToKnee = np.array([hipToHold[0] * math.cos(holdAngle) - hipToHold[1] * math.sin(holdAngle),
                       hipToHold[0] * math.sin(holdAngle) + hipToHold[1] * math.cos(holdAngle)])
  
  # Create a hip-to-shoulder numpy vector as well.
  hipToShoulder = np.array([hip_location[0] - shoulder_location[0], hip_location[1] - shoulder_location[1]])

  # Use numpy to find the angle from shoulder to hip to knee.
  hipAngle = np.arccos(np.dot(hipToShoulder, hipToKnee) / (np.linalg.norm(hipToShoulder) * np.linalg.norm(hipToKnee)))

  return math.degrees(hipAngle)
  
def calcMaxKneeAngle(hipAngle):
  maxKnee = -(6500 * (hipAngle - 90)) ** (1/3) + 112.5
  return maxKnee

def calc_max_hip_angle(kneeAngle):
  # Adjusting maxHip formula to limit maximum hip angle to 90 degrees, let's see if this makes better footholds.
  maxHip = min(90, - (kneeAngle - 112.5) ** 3 / 6500 + 90)
  return maxHip

def foot_check(hold_yMax, position, climber, limb):
  # This function returns True if a hold is reachable with the given foot, and False otherwise.

  max_reach = climber.upper_leg_length + climber.lower_leg_length  # Max reach for legs
  min_reach = math.fabs(climber.upper_leg_length - climber.lower_leg_length)
  
  # A hold is not considered reachable if its distance from the hip is longer than the leg,
  # or it is at or below where the foot already is (going by y-values of coordinates),
  # or if it requires the foot crossing the other side of the body (e.g. right foot is left of left hip),
  # or if it is too close to the hip (distance is less )

  if limb == 'left_foot': 
    # Confirm the foothold is within one leg's length of the hip.
    if distance.euclidean(position.left_hip, hold_yMax) > max_reach: return False
    # Confirm the foothold isn't too close to the hip.
    if distance.euclidean(position.left_hip, hold_yMax) < min_reach: return False
    # Confirm the foothold is above where the foot is currently.
    if position.left_foot[1] - hold_yMax[1] >= 0: return False
    # Confirm the foothold isn't cross-body.
    if hold_yMax[0] > position.right_hip[0]: return False
    # Confirm the foothold isn't already hosting another foot.
    if hold_yMax == position.right_foot: return False

    # If the hold passes the above tests, use trigonometry to determine the necessary knee angle,
    # and then determine if the knee and hip bends required to place a foot on the hold are anatomically possible.
    
    hipAngle = calc_hip_angle(hold_yMax, position.left_hip, position.left_shoulder, climber)
    kneeAngle = calc_knee_angle(hold_yMax, position.left_hip, climber)
    maxHipAngle = calc_max_hip_angle(kneeAngle)
    if hipAngle > maxHipAngle: return False
  
  if limb == 'right_foot':
    # Confirm the foothold is within one leg's length of the hip.
    if distance.euclidean(position.right_hip, hold_yMax) > max_reach: return False
    # Confirm the foothold isn't too close to the hip.
    if distance.euclidean(position.right_hip, hold_yMax) < min_reach: return False
    # Confirm the foothold is above where the foot is currently.
    if position.right_foot[1] - hold_yMax[1] >= 0: return False
    # Confirm the foothold isn't cross-body.
    if hold_yMax[0] > position.left_hip[0]: return False
    # Confirm the foothold isn't already hosting another foot.
    if hold_yMax == position.right_foot: return False

    # If the hold passes the above tests, use trigonometry to determine the necessary knee angle,
    # and then determine if the knee and hip bends required to place a foot on the hold are anatomically possible.

    hipAngle = calc_hip_angle(hold_yMax, position.right_hip, position.right_shoulder, climber)
    kneeAngle = calc_knee_angle(hold_yMax, position.right_hip, climber)
    maxHipAngle = calc_max_hip_angle(kneeAngle)
    if hipAngle > maxHipAngle: return False

  return True