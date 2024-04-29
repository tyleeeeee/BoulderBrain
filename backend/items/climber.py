class Climber:
  def __init__(self, id, wall_id, height=165, upperArmLength=35, forearmLength=25, legLength=70, torsoHeight=70, torsoWidth=44, level=-1):
    self.id = id
    self.wall_id = wall_id
    self.height = height
    self.upperArmLength = upperArmLength
    self.forearmLength = forearmLength
    self.legLength = legLength
    self.torsoHeight = torsoHeight
    self.torsoWidth = torsoWidth
    self.level = level

  def getClimber(self):
    return self

  def setHeight(self, newHeight):
    self.height = newHeight



#def getValidMovesFromPosition(human pose coordinates, valid holds):
  # 1. Check if pose is at the top of the wall, if so, then return none (the route is complete).
  # 2. For each limb:
  #      2.1: Calculate reachable region using limb length and other constraints.
  #      2.2: Find all valid holds within that region.
  # 3. Return the set of valid moves. (Each move is defined by a limb and a hold).


#def getPositionFromMove(human pose coordinates, limb, hold):
  # 1. Update the limb coordinates in the human pose coordinates, using the new limb and new hold.
  # 2. Update the torso coordinates by calling getTorsoFromLimbs.
  # 3. Return the new human pose coordinates.