class Climber:
  def __init__(self, id, wall, height=165, upperArmLength=35, forearmLength=25, legLength=70, torsoHeight=70, torsoWidth=44, level=-1):
    self.id = id
    self.wall = wall
    
    # Climber measurables (floats)
    self.height = height
    self.upperArmLength = upperArmLength
    self.forearmLength = forearmLength
    self.legLength = legLength
    self.torsoHeight = torsoHeight
    self.torsoWidth = torsoWidth

    # Climber skill level (int), default is -1 (VB)
    self.level = level

