import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import LinearConstraint
from backend.items.climber import Climber

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

  def squaredDistances(leftHand, leftShoulder, rightHand, rightShoulder, leftFoot, leftHip, rightFoot, rightHip):
    totalSqDist = cdist(leftHand, leftShoulder) ** 2
    totalSqDist += cdist(rightHand, rightShoulder) ** 2
    totalSqDist += cdist(leftFoot, leftHip) ** 2
    totalSqDist += cdist(rightFoot, rightHip) ** 2
    return totalSqDist
  


  
  # 1. Update the limb coordinates in the human pose coordinates, using the new limb and new hold.
  # 2. Update the torso coordinates by calling getTorsoFromLimbs.
  # 3. Return the new human pose coordinates.