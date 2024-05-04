from .position import Position
from .pose_estimation_service import getPositionFromMove


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
        initialPosition = Position(climber)
        print("entry while loop")
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
    print("Point generateRoutes was reached")
    return finalPositions


def generateRoutesRecursive(climber, wall, position):
    position.timestep += 1
    print("EntryPoint generateRoutesRECURSIVE was reached")
    # Max depth of the tree is 10 moves.
    if position.timestep >= 10:
        print("Max depth of the tree is 10 moves ")
        return []

    # If any hand (or foot) is within 10% of the height of the wall from the top, then declare the
    # route finished.

    if max(position.left_hand[1], position.right_hand[1], position.left_foot[1],
           position.right_foot[1]) >= wall.height * 0.9:
        print("hand/foot is within 10% of the height, so return ")
        return []

    # Array to be returned.
    finalPositions = []

    # If any hand or foot is not yet placed on the wall, then prioritize moving those hands and feet.
    if min(position.left_hand[0], position.right_hand[0], position.left_foot[0], position.right_foot[0]) < 0:

        for limb in ['left_hand', 'right_hand', 'left_foot', 'right_foot']:
            if getattr(position, limb)[0] < 0:
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
    print("ExitPoint generateRoutesRECURSIVE was reached")
    return finalPositions


def getReachableHolds(climber, wall, position, limb):
    reachable_holds = []
    limb_x, limb_y = getattr(position, limb)  # x and y coordinates of a limb

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

        # Check if the hold is within reach
        if distance <= max_reach:
            # Further checks can be added here (e.g., color, size, or specific conditions)
            reachable_holds.append(hold)

    return reachable_holds
