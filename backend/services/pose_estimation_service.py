import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import distance
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
from items.climber import Climber
from items.hold import Hold
from items.position import Position
from items.route import Route
from items.wall import Wall

#def getTorsoFromLimbs(hand/feet positions):
  # 1. Call CIMI4D to estimate body position.
  # 2. Return set of torso location points.

#def getValidMovesFromPosition(human pose coordinates, valid holds):
  # 1. Check if pose is at the top of the wall, if so, then return none (the route is complete).
  # 2. For each limb:
  #      2.1: Calculate reachable region using limb length and other constraints.
  #      2.2: Find all valid holds within that region.
  # 3. Return the set of valid moves. (Each move is defined by a limb and a hold).

def getPositionFromMove(currPosition, climber, newHold, limbToMove):

  newPosition = currPosition
    
  match limbToMove:
    case "leftHand":
      newPosition.left_hand = [newHold.yMax[0], newHold.yMax[1]]
    case "rightHand":
      newPosition.right_hand = [newHold.yMax[0], newHold.yMax[1]]
    case "leftFoot":
      newPosition.left_foot = [newHold.yMax[0], newHold.yMax[1]]
    case "rightFoot":
      newPosition.right_foot = [newHold.yMax[0], newHold.yMax[1]]

  def squaredDistances(position):
    totalSqDist = distance.euclidean(position.left_hand, position.left_shoulder) ** 2
    totalSqDist += distance.euclidean(position.right_hand, position.right_shoulder) ** 2
    totalSqDist += distance.euclidean(position.left_foot, position.left_hip) ** 2
    totalSqDist += distance.euclidean(position.right_foot, position.right_hip) ** 2
    return totalSqDist
  
  # All constraint functions: Hand-shoulder distance cannot exceed arm length, foot-hip distance cannot exceed leg length, 
  # The torso cannot change size or warp.
  def leftLegConstraint(position, climber): 
    return climber. distance.euclidean(position.left_foot, position.left_hip)
  
  def rightLegConstraint(position, climber): 
    return distance.euclidean(position.right_foot, position.right_hip)
  
  def leftArmConstraint(position, climber): # Must be greater than 0 (hand-to-shoulder distance cannot exceed arm length).
    return climber.upper_arm_length + climber.forearm_length - distance.euclidean(position.left_hand, position.left_shoulder)
  
  def rightArmConstraint(position, climber): # Must be greater than 0 (hand-to-shoulder distance cannot exceed arm length).
    return climber.upper_arm_length + climber.forearm_length - distance.euclidean(position.right_hand, position.right_shoulder)
  
  def shoulderWidthConstraint(position, climber): 
    return distance.euclidean(position.left_shoulder, position.right_shoulder)
  def hipWidthConstraint(position, climber): 
    return distance.euclidean(position.left_hip, position.right_hip)
  def leftSideConstraint(position, climber): 
    return distance.euclidean(position.left_hip, position.left_shoulder)
  def rightSideConstraint(position, climber): 
    return distance.euclidean(position.right_hip, position.right_shoulder)
  def crossBodyContraint(position, climber): 
    return distance.euclidean(position.left_shoulder, position.right_hip)

  leftLegEq = {'type':'eq'}
  print(newPosition.left_hand)
  print(leftArmConstraint(newPosition, climber))
  print(rightArmConstraint(newPosition, climber))
  print(leftLegConstraint(newPosition, climber))
  print(rightLegConstraint(newPosition, climber))
  print(squaredDistances(newPosition))
  
newWall = Wall(1, 450, 450)
newClimber = Climber(1, newWall)
newPosition = Position(1, newClimber, 0, 1, [0, 0, 0, 0], [170.0, 170.0], [0.0, 0.0], [200.0, 140.0], [200.0, 70.0], [0.0,0.0], [200.0, 0.0], [275.0, 170.0], [0.0, 0.0], [245.0, 140.0], [245.0, 70.0], [0.0, 0.0], [245.0, 0.0])
newHold = Hold(1, newWall, [0.0, 0.0], "blue", False, [185.0, 155.0])

getPositionFromMove(newPosition, newClimber, newHold, "leftHand")

  
  # 1. Update the limb coordinates in the human pose coordinates, using the new limb and new hold.
  # 2. Update the torso coordinates by calling getTorsoFromLimbs.
  # 3. Return the new human pose coordinates.