import math
from scipy.spatial import distance
import numpy as np
import sympy as sp
from .climber import Climber
from .wall import Wall
from .position import Position

def calc_elbow_location(shoulder_location, hold_location, climber, side="right"):
  
  # Solves for the location of the elbow using the intersection of circles.
  # It's equivalent to solving for the location of a third point of a triangle,
  # when the other two points (the shoulder and the hold/hand) are known,
  # as well as the lengths of all three sides.
  
  a = climber.upper_arm_length
  b = climber.forearm_length
  c = distance.euclidean(hold_location, shoulder_location)

  print("ABC:", a, b, c)

  [x1, y1] = shoulder_location
  [x2, y2] = hold_location

  # Symbols for the unknown coordinates
  x, y = sp.symbols('x y')

  # Equation of the circle centered at the shoulder with radius upper_arm_length
  eq1 = sp.Eq((x - x1)**2 + (y - y1)**2, a**2)

  # Equation of the circle centered at the hold with radius forearm_length
  eq2 = sp.Eq((x - x2)**2 + (y - y2)**2, b**2)

  # Solve the system of equations
  solutions = sp.solve((eq1, eq2), (x, y))

  # There are two possible solutions to the intersection of circles, but we only want one.
  # For now we just take the one with lower y-value, but for future work:
  # Take the vectors from the shoulder to each point, to compare them.
  # If this is the right arm, then we want the point that is further clockwise.
  # If this is the left arm, then we want the point that is further counterclockwise.

  sorted_solutions = sorted(solutions, key=lambda coord:coord[1])
  print(solutions)

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
print("Loading wall and climber details...")
wall = Wall(id=4, height=350, width=450, image_path=image_path)
climber = Climber(wall, height=180, upper_arm_length=35*(9/9), forearm_length=35*(9/9),
                          upper_leg_length=45*(9/9), lower_leg_length=40*(9/9), torso_height=80*(9/9),
                          torso_width=50*(9/9))