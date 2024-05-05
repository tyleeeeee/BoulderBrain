class Climber:
  def __init__(self, wall, height=165, upper_arm_length=35, forearm_length=25, upper_leg_length=35, lower_leg_length=35, torso_height=70, torso_width=45, level=-1):

    self.wall = wall
    
    # Climber measurables (floats)
    self.height = height
    self.upper_arm_length = upper_arm_length
    self.forearm_length = forearm_length
    self.upper_leg_length = upper_leg_length
    self.lower_leg_length = lower_leg_length
    self.torso_height = torso_height
    self.torso_width = torso_width

    # Climber skill level (int), default is -1 (VB)
    self.level = level

