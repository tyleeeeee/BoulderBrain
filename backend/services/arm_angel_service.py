import math
from scipy.spatial import distance
import numpy as np
import sympy as sp
from .climber import Climber
from .wall import Wall
from .position import Position
from scipy.optimize import fsolve

def calc_elbow_location(shoulder_location, hold_location, climber, side="right"):
  
  # Solves for the location of the elbow using the intersection of circles.
  # It's equivalent to solving for the location of a third point of a triangle,
  # when the other two points (the shoulder and the hold/hand) are known,
  # as well as the lengths of all three sides.

  A = hold_location
  B = shoulder_location
  
  a = climber.upper_arm_length
  b = climber.forearm_length
  c = distance.euclidean(hold_location, shoulder_location)

  x1, y1 = A
  x2, y2 = B

  # [x1, y1] = shoulder_location
  # [x2, y2] = hold_location

  # # Symbols for the unknown coordinates
  # x, y = sp.symbols('x y')

  # # Equation of the circle centered at the shoulder with radius upper_arm_length
  # eq1 = sp.Eq((x - x1)**2 + (y - y1)**2, a**2)

  # # Equation of the circle centered at the hold with radius forearm_length
  # eq2 = sp.Eq((x - x2)**2 + (y - y2)**2, b**2)

  # print("About to solve the system of equations.")
  # # Solve the system of equations
  # solutions = sp.solve((eq1, eq2), (x, y))
  # print("Solved the system of equations!")

  # Calculate the angle at A using the law of cosines
  cos_angle_A = (b**2 + c**2 - a**2) / (2 * b * c)

  # Ensure the cosine value is within the valid range [-1, 1] to avoid domain errors
  cos_angle_A = max(min(cos_angle_A, 1), -1)
  # The above line is a ChatGPT-suggested fix and idk if the logic is sound.

  angle_A = np.arccos(cos_angle_A)

  # Calculate the direction vector from A to B
  direction_vector = np.array([x2 - x1, y2 - y1])
  direction_unit_vector = direction_vector / c

  # Calculate the perpendicular vector to the direction vector
  perp_vector = np.array([-direction_unit_vector[1], direction_unit_vector[0]])

  # Calculate the coordinates of the possible third points
  C1 = (np.array(A) + b * np.cos(angle_A) * direction_unit_vector + b * np.sin(angle_A) * perp_vector).tolist()
  C2 = (np.array(A) + b * np.cos(angle_A) * direction_unit_vector - b * np.sin(angle_A) * perp_vector).tolist()

  # There are two possible solutions to the intersection of circles, but we only want one.
  # For now we just take the one with lower y-value, but for future work:
  # Take the vectors from the shoulder to each point, to compare them.
  # If this is the right arm, then we want the point that is further clockwise.
  # If this is the left arm, then we want the point that is further counterclockwise.

  sorted_solutions = sorted([C1, C2], key=lambda coord:coord[1])

  return sorted_solutions[0]

def calc_grip_angle(shoulder_location, hold_location, climber, side="right"):
  elbow_location = calc_elbow_location(shoulder_location, hold_location, climber, side)

  # Find the vector pointing from the hand to the elbow.
  hand2elbow = [elbow_location[0] - hold_location[0], elbow_location[1] - hold_location[1]]

  # Find the angle of the vector relative to the positive x-axis.
  # This is the angle the arm approaches the hold from.

  angle_radians = math.atan2(hand2elbow[1], hand2elbow[0])
  angle_degrees = math.degrees(angle_radians) % 360
  return angle_degrees





image_path = 'services/files/860.jpg'
wall = Wall(id=4, height=350, width=450, image_path=image_path)
climber = Climber(wall, height=180, upper_arm_length=35*(9/9), forearm_length=35*(9/9),
                          upper_leg_length=45*(9/9), lower_leg_length=40*(9/9), torso_height=80*(9/9),
                          torso_width=50*(9/9))

# print(calc_elbow_location([0, 0], [35, 35], climber))