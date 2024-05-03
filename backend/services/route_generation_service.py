from items.climber import Climber
from items.hold import Hold
from items.position import Position
from items.route import Route
from items.wall import Wall

def generateRoutes(wall):
  # 1. Start with an initial position (feet on ground) and an empty queue of states/positions.
  # 2. Add the initial position to the queue.
  # 3. For each position in the queue:
  #     3.1: If the position is a terminal position (at the top of the wall), print the route.
          # How do we want to store the route, as an attribute of the state maybe?
  #     3.2: validMoves = getValidMovesFromPosition(current position, valid holds).
  #     3.3: For every move in validMoves, add getPositionFromMove(move) to queue.
  #     3.4: Pop from queue.

