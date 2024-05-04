from items.climber import Climber
from items.hold import Hold
from items.position import Position
from items.route import Route
from items.wall import Wall
from pose_estimation_service import getPositionFromMove


def generateRoutes(wall, climber):
    # 1. Start with an initial position (feet on ground) and an empty queue of states/positions.
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
        initialPosition = Position(climber)
        initialPosition.timestep = 0

        # Most important for the initial position is the location of the torso, which defines the reachable holds.
        # Hands and feet have negative values to represent that they begin "nowhere" on the wall.

        initialPosition.left_hip = [startPoint - climber.torso_width / 2,
                                    climber.lower_leg_length + climber.upper_leg_length]
        initialPosition.right_hip = [startPoint + climber.torso_width / 2,
                                     climber.lower_leg_length + climber.upper_leg_length]
        initialPosition.left_shoulder = [startPoint - climber.torso_width / 2,
                                         climber.lower_leg_length + climber.upper_leg_length + climber.torso_height]
        initialPosition.right_shoulder = [startPoint + climber.torso_width / 2,
                                          climber.lower_leg_length + climber.upper_leg_length + climber.torso_height]
        initialPosition.left_hand, initialPosition.right_hand = [-1, -1], [-1, -1]
        initialPosition.left_foot, initialPosition.right_foot = [-1, -1], [-1, -1]

        # Explore the full tree of generated routes with generateRoutesRecursive, and append it to the results.
        finalPositions.append(generateRoutesRecursive(climber, wall, initialPosition))

        startPoint += 0.8 * armSpan


def generateRoutesRecursive(climber, wall, position):
    position.timestep += 1

    # Max depth of the tree is 10 moves.
    if position.timestep >= 10: return []

    # If any hand (or foot) is within 10% of the height of the wall from the top, then declare the
    # route finished.

    if max(position.left_hand[1], position.right_hand[1], position.left_foot[1],
           position.right_foot[1]) >= wall.height * 0.9:
        return []

    # Array to be returned.
    finalPositions = []

    # If any hand or foot is not yet placed on the wall, then prioritize moving those hands and feet.
    if min(position.left_hand[0], position.right_hand[0], position.left_foot[0], position.right_foot[0]) < 0:

        for limb in ['left_hand', 'right_hand', 'left_foot', 'right_foot']:
            if position[limb][0] < 0:
                first = 0
                for hold in getReachableHolds(climber, wall, position, limb):
                    newPosition = getPositionFromMove(position, climber, hold, limb)
                    # To control how the tree is pruned, adjust the if statement below.
                    # For now, I'm pruning the tree by only selecting one move per limb
                    # to explore, reducing the branching factor to 4.
                    if (first == 0):
                        finalPositions.append(generateRoutesRecursive(climber, wall, newPosition))
                        first += 1

    # If all four hands and feet are already on the wall, then explore all moves.
    else:
        for limb in ['left_hand', 'right_hand', 'left_foot', 'right_foot']:
            for hold in getReachableHolds(climber, wall, position, limb):
                first = 0
                newPosition = getPositionFromMove(position, climber, hold, limb)
                # To control how the tree is pruned, adjust the if statement below.
                # For now, I'm pruning the tree by only selecting one move per limb
                # to explore, reducing the branching factor to 4.
                if (first == 0):
                    finalPositions.append(generateRoutesRecursive(climber, wall, newPosition))
                    first += 1

    return finalPositions


def getReachableHolds(climber, wall, position, limb):
    return None
