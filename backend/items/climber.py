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

  # Getter and setter for height
  def get_height(self):
    return self.height

  def set_height(self, value):
    self.height = value

  # Getter and setter for upper arm length
  def get_upper_arm_length(self):
    return self.upper_arm_length

  def set_upper_arm_length(self, value):
    self.upper_arm_length = value

  # Getter and setter for forearm length
  def get_forearm_length(self):
    return self.forearm_length

  def set_forearm_length(self, value):
    self.forearm_length = value

  # Getter and setter for leg length
  def get_leg_length(self):
    return self.leg_length

  def set_leg_length(self, value):
    self.leg_length = value

  # Getter and setter for torso height
  def get_torso_height(self):
    return self.torso_height

  def set_torso_height(self, value):
    self.torso_height = value

  # Getter and setter for torso width
  def get_torso_width(self):
    return self.torso_width

  def set_torso_width(self, value):
    self.torso_width = value

  # Getter and setter for level
  def get_level(self):
    return self.level

  def set_level(self, value):
    self.level = value


