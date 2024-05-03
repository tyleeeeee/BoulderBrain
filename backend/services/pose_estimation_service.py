import numpy as np
from scipy.spatial import distance
from scipy.optimize import minimize
from math import sqrt
from items.climber import Climber
from items.hold import Hold
from items.position import Position
from items.route import Route
from items.wall import Wall

#def getValidMovesFromPosition(human pose coordinates, valid holds):
  # 1. Check if pose is at the top of the wall, if so, then return none (the route is complete).
  # 2. For each limb:
  #      2.1: Calculate reachable region using limb length and other constraints.
  #      2.2: Find all valid holds within that region.
  # 3. Return the set of valid moves. (Each move is defined by a limb and a hold).

def getPositionFromMove(oldPosition, climber, newHold, limbToMove):

  currPosition = oldPosition
    
  match limbToMove:
    case "leftHand":
      currPosition.left_hand = [newHold.yMax[0], newHold.yMax[1]]
    case "rightHand":
      currPosition.right_hand = [newHold.yMax[0], newHold.yMax[1]]
    case "leftFoot":
      currPosition.left_foot = [newHold.yMax[0], newHold.yMax[1]]
    case "rightFoot":
      currPosition.right_foot = [newHold.yMax[0], newHold.yMax[1]]
  
  # Vectorize position for optimization function. By index, the values are:
  # x[0]: Left hand x-coordinate
  # x[1]: Left shoulder x-coordinate
  # x[2]: Right hand x-coordinate
  # x[3]: Right shoulder x-coordinate
  # x[4]: Left foot x-coordinate
  # x[5]: Left hip x-coordinate
  # x[6]: Right foot x-coordinate
  # x[7]: Right hip x-coordinate
  # x[8]: Left hand y-coordinate
  # x[9]: Left shoulder y-coordinate
  # x[10]: Right hand y-coordinate
  # x[11]: Right shoulder y-coordinate
  # x[12]: Left foot y-coordinate
  # x[13]: Left hip y-coordinate
  # x[14]: Right foot y-coordinate
  # x[15]: Right hip y-coordinate
  # x[16]: Arm length (forearm length + upper arm length)
  # x[17]: Leg length (lower leg length + upper leg length)
  # x[18]: Torso height
  # x[19]: Torso width


  # Objective function for scipy.minimize.
  def squaredDistances(x):
    totalSqDist = distance.euclidean([x[0], x[8]], [x[1], x[9]]) ** 2
    totalSqDist += distance.euclidean([x[2], x[10]], [x[3], x[11]]) ** 2
    totalSqDist += distance.euclidean([x[4], x[12]], [x[5], x[13]]) ** 2
    totalSqDist += distance.euclidean([x[6], x[14]], [x[7], x[15]]) ** 2
    return totalSqDist

  # All constraint functions: 
  # Hand-shoulder distance cannot exceed arm length (x[16]), 
  # Foot-hip distance cannot exceed leg length (x[17]), 
  # The torso cannot change size (four fixed distances between adjacent corners of the rectangle),
  # The torso cannot warp/shear (one fixed "seatbelt" distance between two opposite corners).

  # Must be greater than 0 (hand-to-shoulder distance cannot exceed arm length).
  def leftArmConstraint(x): return x[16] - distance.euclidean([x[1], x[9]], [x[2], x[10]])
  
  # Must be greater than 0 (hand-to-shoulder distance cannot exceed arm length).
  def rightArmConstraint(x): return x[16] - distance.euclidean([x[2], x[10]], [x[3], x[11]])

  # Must be greater than 0 (foot-to-hip distance cannot exceed leg length).
  def leftLegConstraint(x): return x[17] - distance.euclidean([x[4], x[12]], [x[5], x[13]])
  
  # Must be greater than 0 (foot-to-hip distance cannot exceed leg length).
  def rightLegConstraint(x): return x[17] - distance.euclidean([x[6], x[14]], [x[7], x[15]])
  
  # Must equal 0 (distance between shoulders is fixed).
  def shoulderWidthConstraint(x): return x[19] - distance.euclidean([x[1], x[9]], [x[3], x[11]])
  
  # Must equal 0 (distance between hips is fixed).
  def hipWidthConstraint(x): return x[19] - distance.euclidean([x[5], x[13]], [x[7], x[15]])
  
  # Must equal 0 (distance between shoulder and hip is fixed).
  def leftSideConstraint(x): return x[18] - distance.euclidean([x[1], x[9]], [x[5], x[13]])
  
  # Must equal 0 (distance between shoulder and hip is fixed).
  def rightSideConstraint(x): return x[18] - distance.euclidean([x[3], x[11]], [x[7], x[15]])
  
  # Must equal 0 ("seatbelt" distance between left shoulder and right hip is fixed, no shearing of torso).
  def crossBodyContraint(x): return sqrt(x[18] ** 2 + x[19] ** 2) - distance.euclidean([x[1], x[9]], [x[7], x[15]])

  # Constraint dictionairies for scipy.minimize.
  leftLegIneq = {'type':'ineq', 'fun': leftLegConstraint}
  rightLegIneq = {'type':'ineq', 'fun': rightLegConstraint}
  leftArmIneq = {'type':'ineq', 'fun': leftArmConstraint}
  rightArmIneq = {'type':'ineq', 'fun': rightArmConstraint}
  shoulderEq = {'type': 'eq', 'fun': shoulderWidthConstraint}
  hipEq = {'type': 'eq', 'fun': hipWidthConstraint}
  leftSideEq = {'type': 'eq', 'fun': leftSideConstraint}
  rightSideEq = {'type': 'eq', 'fun': rightSideConstraint}
  crossBodyEq = {'type': 'eq', 'fun': crossBodyContraint}

  # Initial guess for scipy.minimize is just the current position.
  x0 = np.array([
    currPosition.left_hand[0], 
    currPosition.left_shoulder[0], 
    currPosition.right_hand[0], 
    currPosition.right_shoulder[0], 
    currPosition.left_foot[0], 
    currPosition.left_hip[0], 
    currPosition.right_foot[0], 
    currPosition.right_hip[0], 
    currPosition.left_hand[1], 
    currPosition.left_shoulder[1], 
    currPosition.right_hand[1], 
    currPosition.right_shoulder[1], 
    currPosition.left_foot[1], 
    currPosition.left_hip[1], 
    currPosition.right_foot[1], 
    currPosition.right_hip[1],
    climber.upper_arm_length + climber.forearm_length,
    climber.lower_leg_length + climber.upper_leg_length,
    climber.torso_height,
    climber.torso_width
  ])   

  # Use bounds to fix x- and y- coordinates of hands and feet to their current values. 
  # For other variables, set bounds to 'None'.

  bounds = [
    # 0: Left hand x-coordinate fixed.
    (currPosition.left_hand[0], currPosition.left_hand[0]),
    # 1: Left shoulder x-coordinate unfixed.
    (None, None),
    # 2: Right hand x-coordinate fixed.
    (currPosition.right_hand[0], currPosition.right_hand[0]),
    # 3: Right shoulder x-coordinate unfixed.
    (None, None),
    # 4: Left foot x-coordinate fixed.
    (currPosition.left_foot[0], currPosition.left_foot[0]),
    # 5: Left hip x-coordinate unfixed.
    (None, None),
    # 6: Right foot x-coordinate fixed.
    (currPosition.right_foot[0], currPosition.right_foot[0]),
    # 7: Right hip x-coordinate unfixed.
    (None, None),
    # 8: Left hand y-coordinate fixed.
    (currPosition.left_hand[1], currPosition.left_hand[1]),
    # 9: Left shoulder y-coordinate unfixed.
    (None, None),
    # 10: Right hand y-coordinate fixed.
    (currPosition.right_hand[1], currPosition.right_hand[1]),
    # 11: Right shoulder y-coordinate unfixed.
    (None, None),
    # 12: Left foot y-coordinate fixed.
    (currPosition.left_foot[1], currPosition.left_foot[1]),
    # 13: Left hip y-coordinate unfixed.
    (None, None),
    # 14: Right foot y-coordinate fixed.
    (currPosition.right_foot[1], currPosition.right_foot[1]),
    # 15: Right hip y-coordinate unfixed.
    (None, None),
    # 16, 17, 18, 19: Fixed distances between unfixed points are controlled via constraints, so there are no bounds.
    (None, None),
    (None, None),
    (None, None),
    (None, None)
  ]

  # Run scipy.minimize.
  result_eq = minimize(squaredDistances, x0, bounds = bounds, constraints = [leftLegIneq, rightLegIneq, leftArmIneq, rightArmIneq, shoulderEq, hipEq, leftSideEq, rightSideEq, crossBodyEq])

  # Update the position using outputs from scipy.minimuze.
  currPosition.left_shoulder = [result_eq.x[1], result_eq.x[9]]
  currPosition.right_shoulder = [result_eq.x[3], result_eq.x[11]]
  currPosition.left_hip = [result_eq.x[5], result_eq.x[13]]
  currPosition.right_hip = [result_eq.x[7], result_eq.x[15]]
  
  # Return the updated position.
  return currPosition




newWall = Wall(1, 450, 450)
newClimber = Climber(1, newWall)
newPosition = Position(1, newClimber, 0, 1, [0, 0, 0, 0], [170.0, 170.0], [0.0, 0.0], [200.0, 140.0], [200.0, 70.0], [0.0,0.0], [200.0, 0.0], [275.0, 170.0], [0.0, 0.0], [245.0, 140.0], [245.0, 70.0], [0.0, 0.0], [245.0, 0.0])
newHold = Hold(1, newWall, [0.0, 0.0], "blue", False, [185.0, 155.0])

updatedPosition = getPositionFromMove(newPosition, newClimber, newHold, "leftHand")
  
  # 1. Update the limb coordinates in the human pose coordinates, using the new limb and new hold.
  # 2. Update the torso coordinates by calling getTorsoFromLimbs.
  # 3. Return the new human pose coordinates.


  # def vectorToString(x):
  #   print("Left hand:", (x[0], x[8]))
  #   print("Left shoulder:", (x[1], x[9]))
  #   print("Right hand:", (x[2], x[10]))
  #   print("Right shoulder:", (x[3], x[11]))
  #   print("Left foot:", (x[4], x[12]))
  #   print("Left hip:", (x[5], x[13]))
  #   print("Right foot:", (x[6], x[14]))
  #   print("Right hip:", (x[7], x[15]))
  #   print("Shoulder width:", x[19])
  #   print("Shoulder distance:", distance.euclidean([x[1], x[9]], [x[3], x[11]]))
  #   print("Hip width:", x[19])
  #   print("Hip distance:", distance.euclidean([x[5], x[13]], [x[7], x[15]]))
  #   print("Left side height:", x[18])
  #   print("Left side distance:", distance.euclidean([x[1], x[9]], [x[5], x[13]]))
  #   print("Right side height:", x[18])
  #   print("Right side distance:", distance.euclidean([x[3], x[11]], [x[7], x[15]]))
  #   print("Cross-body height:", sqrt(x[18] ** 2 + x[19] ** 2))
  #   print("Cross-body distance:", distance.euclidean([x[1], x[9]], [x[7], x[15]]))

  # print("Original:")
  # vectorToString(x0)
  # print("Minimized:")
  # vectorToString(result_eq.x)