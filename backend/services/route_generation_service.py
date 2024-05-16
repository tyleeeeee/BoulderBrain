
from .pose_estimation_service import getPositionFromMove
from .image_processing_service import generate_dense_holds, get_holds_from_image
from .reachable_foot_area import foot_check
from scipy.spatial import distance
from .position import Position
from .wall import Wall
from .climber import Climber
import copy
import random
import math
import re


# need to define inital position of limbs to calculate distance in reachable area. Setting them to  [-1, -1] means they start at an undefined position on the wall
def initializePosition(climber, startPoint, wall):
  initialPosition = Position(climber)
  # baseHeight = climber.lower_leg_length + 10 # TODO discuss assumption that climber start with feet at this height

  # Set initial limb positions
  initialPosition.left_hip = [startPoint - climber.torso_width / 2,
                              climber.lower_leg_length + climber.upper_leg_length]
  initialPosition.right_hip = [startPoint + climber.torso_width / 2,
                                climber.lower_leg_length + climber.upper_leg_length]
  initialPosition.left_shoulder = [startPoint - climber.torso_width / 2,
                                    climber.lower_leg_length + climber.upper_leg_length + climber.torso_height]
  initialPosition.right_shoulder = [startPoint + climber.torso_width / 2,
                                    climber.lower_leg_length + climber.upper_leg_length + climber.torso_height]
  initialPosition.left_hand, initialPosition.right_hand = initialPosition.left_shoulder, initialPosition.right_shoulder
  initialPosition.left_foot, initialPosition.right_foot = [startPoint - climber.torso_width / 2, 0], [startPoint + climber.torso_width / 2, 0]

  return initialPosition

#From current position, selects four moves, one for each limb, prioritizing the biggest vertical moves.
def selectNextMoves(climber, wall, current_position):
  best_moves = []

  # If any limbs are not yet placed on the wall, then prioritize placing them on the wall.
  if min(current_position.left_hand[0], 
         current_position.right_hand[0], 
         current_position.left_foot[0], 
         current_position.right_foot[0]) < 0:
    
    for limb in ['left_hand', 'right_hand', 'left_foot', 'right_foot']:
      if getattr(current_position, limb)[0] < 0:
        reachable_holds = getReachableHolds(climber, wall, current_position, limb)

        if reachable_holds:
          # Sort moves by height/highest y-value.
          reachable_holds.sort(key = lambda hold: getattr(hold, 'yMax')[1], reverse=True)
        
          highest_hold = reachable_holds[0]
          #print("Currently, the", limb, "is at", getattr(current_position, limb), "and the hold it's moving to is at", highest_hold.yMax)
          newPosition = getPositionFromMove(current_position, climber, highest_hold, limb)

          # if newPosition == current_position: print("getPosition returned the same position!")
          # else: print("getPosition updated the position!")
          
          best_moves.append(newPosition)
    
            

  else:
    for limb in ['left_hand', 'right_hand', 'left_foot', 'right_foot']:

      reachable_holds = getReachableHolds(climber, wall, current_position, limb)
      if reachable_holds:
        
        # Sort moves by height/highest y-value.
        reachable_holds.sort(key = lambda hold: getattr(hold, "yMax")[1], reverse=True)
        highest_hold = reachable_holds[0]
        #print("Currently, the", limb, "is at", getattr(current_position, limb), "and the hold it's moving to is at", highest_hold.yMax)
        newPosition = getPositionFromMove(current_position, climber, highest_hold, limb)
        # if newPosition == current_position:
        #    print("getPosition returned the same position!")
        # else: print("getPosition updated the position!")

        best_moves.append(newPosition)

  if best_moves: 
     print("Number of moves we explore next:", len(best_moves))
     return best_moves

  else:
    print("No best moves found.")
    return best_moves


def generateRoutes(wall, climber):
    # 1. Start with an initial position (feet on ground) and an empty queue of states/positions. #TODO: can this be deleted here? Queues ar enot implemenetd here
    # 2. Add the initial position to the queue.
    # 3. For each position in the queue:
    #     3.1: If the position is a terminal position (at the top of the wall), print the route.
    # How do we want to store the route, as an attribute of the state maybe?
    #     3.2: validMoves = getValidMovesFromPosition(current position, valid holds).
    #     3.3: For every move in validMoves, add getPositionFromMove(move) to queue.
    #     3.4: Pop from queue

    # "Plant a tree root" every few meters, specifically every 80% of the climber's arm span,
    # Each site defines where the climber originates in a BFS exploration of possible routes.
    # All holds that are reachable from the ground should be reachable from at least one initial position.

    startPoint = 0
    armSpan = (climber.upper_arm_length + climber.forearm_length) * 2 + climber.torso_width
    startPoint += armSpan / 2

    # finalPositions is an array of all final positions, one per generated route.
    # Each position points to its parent position, which can be used to create
    # in reverse an ordered list of positions to store the full route .
    finalPositions = []

    while startPoint < wall.width:
        # The starting point for
        initialPosition =  initializePosition(climber, startPoint, wall)
        print("Tree root (route starting point): ", startPoint)
        initialPosition.timestep = 0

        # Most important for the initial position is the location of the torso, which defines the reachable holds.
        # Hands and feet have negative values to represent that they begin "nowhere" on the wall.

        # Explore the full tree of generated routes with generateRoutesRecursive, and append it to the results.
        finalPositions = finalPositions + generateRoutesRecursive(climber, wall, initialPosition, None)

        startPoint += 0.8 * armSpan
    return finalPositions


def generateRoutesRecursive(climber, wall, position, parentPosition):
    if (parentPosition != None and position.toString() == parentPosition.toString()): print("Error! Position's parent is itself!!!!!!!!!!!!!!!!!!!!!")
    position.parent_position = parentPosition
    position.timestep += 1
    # Max depth of the tree is 30 moves.
    if position.timestep >= 6:
        #print("Max depth of the tree is 8 moves.")
        position.climber = None

        return [position]

    # If any hand (or foot) is within 10% of the height of the wall from the top, then declare the
    # route finished.

    if max(position.left_hand[1], position.right_hand[1], position.left_foot[1],
           position.right_foot[1]) >= wall.height * 0.9:
        print("Hand/foot is within 10% of the height from the top of the wall, so the route is finished. ")
        position.climber = None
        return [position]

    # Array to be returned.
    finalPositions = [position]

    # If any limbs are not placed on the wall already, prioritize finding moves for them.
    if min(position.left_hand[0], position.right_hand[0], position.left_foot[0], position.right_foot[0]) < 0:

      # Returns an array of positions.
      nextMoves = selectNextMoves(climber, wall, position)

      for nextPosition in nextMoves:
        # getPositionFromMove(current_position, climber, highest_hold, limb)
        if (nextPosition != position):
          finalPositions = finalPositions + generateRoutesRecursive(climber, wall, nextPosition, position)
        else: print("All the found positions equal the current position!")
      # else: print("Alert: No best move could be selected based on the current criteria.")

    else:
      # If all limbs are already on the wall, explore moves for all limbs.
      nextMoves = selectNextMoves(climber, wall, position)

      for nextPosition in nextMoves:
        # getPositionFromMove(current_position, climber, highest_hold, limb)
        if (nextPosition != position):
          #print("These positions are not equal!")
          finalPositions = finalPositions + (generateRoutesRecursive(climber, wall, nextPosition, position))
        #else: print("All the found positions equal the current position!")
      # else: print("Alert: No best move could be selected based on the current criteria.")

        # move_found = False
        # for limb in ['left_hand', 'right_hand', 'left_foot', 'right_foot']:
        #     for hold in getReachableHolds(climber, wall, position, limb):
        #         newPosition = getPositionFromMove(position, climber)
        #         if newPosition:
        #             finalPositions += generateRoutesRecursive(climber, wall, newPosition)
        #             move_found = True
        #             break  # for now: we break after first successful move to reduce complexity
        #     if move_found:
        #         break

    # handle case when no moves are possible
    if not finalPositions:
        print("No further moves possible from this position.")

    return finalPositions


def getReachableHolds(climber, wall, position, limb):
  
  reachable_holds = []

  # limb_x, limb_y = getattr(position, limb)  # x and y coordinates of a limb

  if limb == "left_hand": 
    limb_x, limb_y = position.left_shoulder
    currentHold = position.left_hand
  if limb == "right_hand": 
    limb_x, limb_y = position.right_shoulder
    currentHold = position.right_hand
  if limb == "left_foot": 
    limb_x, limb_y = position.left_hip
    currentHold = position.left_foot
  if limb == "right_foot": 
    limb_x, limb_y = position.right_hip
    currentHold = position.right_foot

  # If the limb is a hand, check all holds to see if they are in reach.
  if 'hand' in limb:
    max_reach = climber.upper_arm_length + climber.forearm_length  # Max reach for hands

    # Iterate through all holds on the wall
    for hold in wall.holds:
      hold_x, hold_y = hold.yMax  # Location of the hold on the wall

      # Calculate distance from the current limb position to the hold
      distance = ((hold_x - limb_x) ** 2 + (hold_y - limb_y) ** 2) ** 0.5

      # print(f"Checking hold at ({hold_x}, {hold_y}) from limb at ({limb_x}, {limb_y}) with distance {distance} and max reach {max_reach}")

      # Check if the hold is within reach, and not the same hold the hand is on, and that its y coordinate is higher than the current holds.
      if distance <= max_reach and currentHold != [hold_x, hold_y] and hold_y > currentHold[1]:
        reachable_holds.append(hold)

  # If limb is a foot, use foot_check to determine if a hold is reachable.
  else:
    for hold in wall.holds:
      if foot_check(hold.yMax, position, climber, limb): reachable_holds.append(hold)


  # if not reachable_holds:
  #   print("No reachable holds found for the", limb, ".")
  # else:
  #   print("Reachable holds are available:", len(reachable_holds))

  return reachable_holds


#TODO: delete/move to app.py??
wall = Wall(id=1, height=400, width=500) #made it quite larger on purpose
climber = Climber(wall, height=180, upper_arm_length=40, forearm_length=30,
                          upper_leg_length=45, lower_leg_length=40, torso_height=80,
                          torso_width=50)

# Set up a new wall with holds
wall.holds = get_holds_from_image()


#new wall with dense holds
wall.holds = generate_dense_holds(wall)


# new function to select final routes

def process_final_routes(routes):
    holds_dict = {}
    routes_description_dict = {}

    for i, route in enumerate(routes):
        route_key = f"route{i + 1}"  # Unique key for each route
        holds_set = set()
        route_description = []
        current_position = route  # Start with the final position in the route
        iteration = 0

        while current_position.parent_position is not None:
            iteration += 1
            iteration_desc = {
                "Iteration": iteration,
                "Current Position": current_position.toString(),
                "Parent Position": current_position.parent_position.toString()
            }
            route_description.append(iteration_desc)

            # Extract hold positions from current and parent positions and update the set
            holds_set.update(extract_holds(current_position))
            holds_set.update(extract_holds(current_position.parent_position))

            # Move to the parent position to continue the traversal
            current_position = current_position.parent_position

        # Store data in dicts
        holds_dict[route_key] = ", ".join(map(str, holds_set))
        routes_description_dict[route_key] = list(reversed(route_description)) # TODO: reverse it?

    return holds_dict, routes_description_dict

def extract_holds(position):
    # Extract hold coordinates from our position object
    holds = {
        (position.left_hand[0], position.left_hand[1]),
        (position.right_hand[0], position.right_hand[1]),
        (position.left_foot[0], position.left_foot[1]),
        (position.right_foot[0], position.right_foot[1])
    }
    return holds

###########################################################
# generate routes
# TODO: this is where it actually gets called --> need to be moved to app.py
routes = generateRoutes(wall, climber)

print("Number of routes generated: ", len(routes))

holds_dict, routes_description_dict = process_final_routes(routes)
#print(holds_dict)
#print(routes_description_dict)



# # we select random route to show
# finalPosition = routes[random.randint(1, len(routes))]
# finalRoute = [finalPosition.toString()]
#
# # searches through linked structure to get nicely structured output
# currentPosition = finalPosition
# parentPosition = currentPosition.parent_position
# iteration = 0
# while (currentPosition.parent_position != None):
#   iteration += 1
#   print("Iteration: ", iteration)
#   print("Current position:", currentPosition.toString())
#   print("Parent position: ", parentPosition.toString())
#   finalRoute.insert(0, currentPosition.parent_position.toString())
#   currentPosition = copy.deepcopy(parentPosition)
#   parentPosition = currentPosition.parent_position

# print(finalRoute)