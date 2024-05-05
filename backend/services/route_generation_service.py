from position import Position
from pose_estimation_service import getPositionFromMove
import json
from image_processing_service import generate_dense_holds, get_holds_from_image
from wall import Wall
from climber import Climber
# import matplotlib as plt

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
          reachable_holds.sort(key = lambda hold: getattr(hold, 'location')[1], reverse=True)
        
          highest_hold = reachable_holds[0]
          newPosition = getPositionFromMove(current_position, climber, highest_hold, limb)
          
          best_moves.append(newPosition)
    
            

  else:
    for limb in ['left_hand', 'right_hand', 'left_foot', 'right_foot']:

      reachable_holds = getReachableHolds(climber, wall, current_position, limb)
      if reachable_holds:
        
        # Sort moves by height/highest y-value.
        reachable_holds.sort(key = lambda hold: getattr(hold, "location")[1], reverse=True)
        highest_hold = reachable_holds[0]
        newPosition = getPositionFromMove(current_position, climber, highest_hold, limb)

        best_moves.append(newPosition)

  if best_moves: return best_moves

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
        print("entry while loop with starting point: ", startPoint)
        initialPosition.timestep = 0

        # Most important for the initial position is the location of the torso, which defines the reachable holds.
        # Hands and feet have negative values to represent that they begin "nowhere" on the wall.

        # Explore the full tree of generated routes with generateRoutesRecursive, and append it to the results.
        finalPositions.append(generateRoutesRecursive(climber, wall, initialPosition))

        startPoint += 0.8 * armSpan
    return finalPositions


def generateRoutesRecursive(climber, wall, position):
    position.timestep += 1
    # Max depth of the tree is 30 moves.
    if position.timestep >= 30:
        print("Max depth of the tree is 30 moves ")
        position.climber = None

        toReturn = json.dumps(position.__dict__)
        return [toReturn]

    # If any hand (or foot) is within 10% of the height of the wall from the top, then declare the
    # route finished.

    if max(position.left_hand[1], position.right_hand[1], position.left_foot[1],
           position.right_foot[1]) >= wall.height * 0.9:
        print("hand/foot is within 10% of the height from the top of the wall, so the route is finished. ")
        position.climber = None
        toReturn = json.dumps(position.__dict__)
        return [toReturn]

    # Array to be returned.
    finalPositions = []

    # If any limbs are not placed on the wall already, prioritize finding moves for them.
    if min(position.left_hand[0], position.right_hand[0], position.left_foot[0], position.right_foot[0]) < 0:
      for position in selectNextMoves(climber, wall, position):
        # getPositionFromMove(current_position, climber, highest_hold, limb)
        finalPositions += generateRoutesRecursive(climber, wall, position)
      else:
          print("Alert: No best move could be selected based on the current criteria.")

    else:
      # If all limbs are already on the wall, explore moves for all limbs.
      for position in selectNextMoves(climber, wall, position):
        # getPositionFromMove(current_position, climber, highest_hold, limb)
        finalPositions += generateRoutesRecursive(climber, wall, position)
      else:
          print("Alert: No best move could be selected based on the current criteria.")
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

    if limb == "left_hand": limb_x, limb_y = position.left_shoulder
    if limb == "right_hand": limb_x, limb_y = position.right_shoulder
    if limb == "left_foot": limb_x, limb_y = position.left_hip
    if limb == "right_foot": limb_x, limb_y = position.right_hip

    # Define reachability limits based on limb type
    if 'hand' in limb:
        max_reach = climber.upper_arm_length + climber.forearm_length  # Max reach for hands
    else:
        max_reach = climber.upper_leg_length + climber.lower_leg_length  # Max reach for feet

    # Iterate through all holds on the wall
    for hold in wall.holds:
        hold_x, hold_y = hold.location  # Location of the hold on the wall

        # Calculate distance from the current limb position to the hold
        distance = ((hold_x - limb_x) ** 2 + (hold_y - limb_y) ** 2) ** 0.5

        print(
            f"Checking hold at ({hold_x}, {hold_y}) from limb at ({limb_x}, {limb_y}) with distance {distance} and max reach {max_reach}")

        # Check if the hold is within reach
        if distance <= max_reach:
            reachable_holds.append(hold)

    if not reachable_holds:
        print("No reachable holds found for this limb.")
    else:
        print("Reachable holds are available.")

    return reachable_holds

wall = Wall(id=1, height=400, width=500) #made it quite larger on purpose
climber = Climber(wall, height=180, upper_arm_length=40, forearm_length=30,
                          upper_leg_length=45, lower_leg_length=40, torso_height=80,
                          torso_width=50)

# # Set up a new wall with holds
# wall.holds = get_holds_from_image()
#

#new wall with dense holds
wall.holds = get_holds_from_image()

# generate routes
routes = generateRoutes(wall, climber)

print(routes)