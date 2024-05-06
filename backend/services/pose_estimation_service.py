import numpy as np
from scipy.spatial import distance
from scipy.optimize import minimize
from math import sqrt
from climber import Climber
from hold import Hold
from position import Position
from wall import Wall
import copy

def vectorToString(x):
  print("Left hand:", (x[0], x[8]))
  print("Left shoulder:", (x[1], x[9]))
  print("Right hand:", (x[2], x[10]))
  print("Right shoulder:", (x[3], x[11]))
  print("Left foot:", (x[4], x[12]))
  print("Left hip:", (x[5], x[13]))
  print("Right foot:", (x[6], x[14]))
  print("Right hip:", (x[7], x[15]))
  print("Shoulder width:", x[19])
  print("Shoulder distance:", distance.euclidean([x[1], x[9]], [x[3], x[11]]))
  print("Hip width:", x[19])
  print("Hip distance:", distance.euclidean([x[5], x[13]], [x[7], x[15]]))
  print("Left side height:", x[18])
  print("Left side distance:", distance.euclidean([x[1], x[9]], [x[5], x[13]]))
  print("Right side height:", x[18])
  print("Right side distance:", distance.euclidean([x[3], x[11]], [x[7], x[15]]))
  print("Cross-body height:", sqrt(x[18] ** 2 + x[19] ** 2))
  print("Cross-body distance:", distance.euclidean([x[1], x[9]], [x[7], x[15]]))


def getPositionFromMove(oldPosition, climber, newHold, limbToMove):
    currPosition = copy.deepcopy(oldPosition)

    if limbToMove == "left_hand":
        print("Updating left hand from", currPosition.left_hand, "to", [newHold.yMax[0], newHold.yMax[1]])
        currPosition.left_hand = [newHold.yMax[0], newHold.yMax[1]]
    elif limbToMove == "right_hand":
        print("Updating right hand from", currPosition.right_hand, "to", [newHold.yMax[0], newHold.yMax[1]])
        currPosition.right_hand = [newHold.yMax[0], newHold.yMax[1]]
    elif limbToMove == "left_foot":
        print("Updating left foot from", currPosition.left_foot, "to", [newHold.yMax[0], newHold.yMax[1]])
        currPosition.left_foot = [newHold.yMax[0], newHold.yMax[1]]
    elif limbToMove == "right_foot":
        print("Updating right foot from", currPosition.right_foot, "to", [newHold.yMax[0], newHold.yMax[1]])
        currPosition.right_foot = [newHold.yMax[0], newHold.yMax[1]]
    else: print("Error: limbToMove is invalid.")

    if currPosition == oldPosition: print("Current still equals old somehow.")
    # Vectorize position for optimization function (accepts 1D array). By index, the values are:
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
        # print(totalSqDist)
        return totalSqDist

    # All constraint functions:
    # Hand-shoulder distance cannot exceed arm length (x[16]),
    # Foot-hip distance cannot exceed leg length (x[17]),
    # The torso cannot change size (four fixed distances between adjacent corners of the rectangle),
    # The torso cannot warp/shear (one fixed "seatbelt" distance between two opposite corners).

    # Must be greater than 0 (hand-to-shoulder distance cannot exceed arm length).
    def leftArmConstraint(x):
        return x[16] - distance.euclidean([x[1], x[9]], [x[2], x[10]])

    # Must be greater than 0 (hand-to-shoulder distance cannot exceed arm length).
    def rightArmConstraint(x):
        return x[16] - distance.euclidean([x[2], x[10]], [x[3], x[11]])

    # Must be greater than 0 (foot-to-hip distance cannot exceed leg length).
    def leftLegConstraint(x):
        return x[17] - distance.euclidean([x[4], x[12]], [x[5], x[13]])

    # Must be greater than 0 (foot-to-hip distance cannot exceed leg length).
    def rightLegConstraint(x):
        return x[17] - distance.euclidean([x[6], x[14]], [x[7], x[15]])

    # Must equal 0 (distance between shoulders is fixed).
    def shoulderWidthConstraint(x):
        return x[19] - distance.euclidean([x[1], x[9]], [x[3], x[11]])

    # Must equal 0 (distance between hips is fixed).
    def hipWidthConstraint(x):
        return x[19] - distance.euclidean([x[5], x[13]], [x[7], x[15]])

    # Must equal 0 (distance between shoulder and hip is fixed).
    def leftSideConstraint(x):
        return x[18] - distance.euclidean([x[1], x[9]], [x[5], x[13]])

    # Must equal 0 (distance between shoulder and hip is fixed).
    def rightSideConstraint(x):
        return x[18] - distance.euclidean([x[3], x[11]], [x[7], x[15]])

    # Must equal 0 ("seatbelt" distance between left shoulder and right hip is fixed, no shearing of torso).
    def crossBodyContraint(x):
        return sqrt(x[18] ** 2 + x[19] ** 2) - distance.euclidean([x[1], x[9]], [x[7], x[15]])

    # Constraint dictionairies for scipy.minimize.
    leftLegIneq = {'type': 'ineq', 'fun': leftLegConstraint}
    rightLegIneq = {'type': 'ineq', 'fun': rightLegConstraint}
    leftArmIneq = {'type': 'ineq', 'fun': leftArmConstraint}
    rightArmIneq = {'type': 'ineq', 'fun': rightArmConstraint}
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
        (climber.upper_arm_length + climber.forearm_length, climber.upper_arm_length + climber.forearm_length),
        (climber.lower_leg_length + climber.upper_leg_length, climber.lower_leg_length + climber.upper_leg_length),
        (climber.torso_height, climber.torso_height),
        (climber.torso_width, climber.torso_width)
    ]

    # Run scipy.minimize.
    result_eq = minimize(squaredDistances, x0, bounds=bounds,
                         constraints=[leftLegIneq, rightLegIneq, leftArmIneq, rightArmIneq, 
                                      shoulderEq, hipEq,
                                      leftSideEq, rightSideEq, 
                                      crossBodyEq])

    # Update the position using outputs from scipy.minimuze.
    currPosition.left_shoulder = [result_eq.x[1], result_eq.x[9]]
    currPosition.right_shoulder = [result_eq.x[3], result_eq.x[11]]
    currPosition.left_hip = [result_eq.x[5], result_eq.x[13]]
    currPosition.right_hip = [result_eq.x[7], result_eq.x[15]]

    

    # print("After moving the limb:")
    # vectorToString(x0)
    # print("After adjusting the torso:")
    # vectorToString(result_eq.x)

    # Return the updated position.
    return currPosition


# TODO: can this be deleted?
# newWall = Wall(1, 450, 450)
# newClimber = Climber(newWall)
# newPosition = Position(newClimber, 0, 1, [0, 0, 0, 0], 
#                        [170.0, 170.0], # left hand
#                        [0.0, 0.0], # left elbow
#                        [200.0, 140.0], #left shoulder
#                        [200.0, 70.0], #left hip
#                        [0.0, 0.0], #left knee
#                        [200.0, 0.0], # left foot
#                        [275.0, 170.0], # right hand, and so on...
#                        [0.0, 0.0], 
#                        [245.0, 140.0],
#                        [245.0, 70.0], 
#                        [0.0, 0.0],
#                        [245.0, 0.0])

# newHold1 = Hold(newWall, [0.0, 0.0], "blue", False, [185.0, 185.0])
# newHold2 = Hold(newWall, [0.0, 0.0], "blue", False, [275.0, 200.0])
# newHold3 = Hold(newWall, [0.0, 0.0], "blue", False, [200.0, 20.0])
# newHold4 = Hold(newWall, [0.0, 0.0], "blue", False, [245.0, 50.0])

# updatedPosition1 = getPositionFromMove(newPosition, newClimber, newHold1, "leftHand")
# updatedPosition2 = getPositionFromMove(updatedPosition1, newClimber, newHold2, "rightHand")
# updatedPosition3 = getPositionFromMove(updatedPosition1, newClimber, newHold3, "leftFoot")
# updatedPosition4 = getPositionFromMove(updatedPosition1, newClimber, newHold4, "rightFoot")

# 1. Update the limb coordinates in the human pose coordinates, using the new limb and new hold.
# 2. Update the torso coordinates by calling getTorsoFromLimbs.
# 3. Return the new human pose coordinates.
