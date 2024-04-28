#def getTorsoFromLimbs(hand/feet positions):
  # 1. Call CIMI4D to estimate body position.
  # 2. Return set of torso location points.




#def getValidMovesFromPosition(human pose coordinates, valid holds):
  # 1. Check if pose is at the top of the wall, if so, then return none (the route is complete).
  # 2. For each limb:
  #      2.1: Calculate reachable region using limb length and other constraints.
  #      2.2: Find all valid holds within that region.
  # 3. Return the set of valid moves. (Each move is defined by a limb and a hold).

