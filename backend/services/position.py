class Position:
    def __init__(self, climber, timestep = 0, parent_position = None, holds = [None, None, None, None], left_hand = [0.0, 0.0], left_elbow = [0.0, 0.0], left_shoulder = [0.0, 0.0], left_hip = [0.0, 0.0], left_knee = [0.0, 0.0], left_foot = [0.0, 0.0], right_hand = [0.0, 0.0], right_elbow = [0.0, 0.0], right_shoulder = [0.0, 0.0], right_hip = [0.0, 0.0], right_knee = [0.0, 0.0], right_foot = [0.0, 0.0], reachable_holds_left_hand = [], reachable_holds_right_hand = [], reachable_holds_left_foot = [], reachable_holds_right_foot = [], reachable_positions = [], start = False, end = False):
        self.timestep = timestep  # depth in search tree
        self.climber_id = climber
        self.parent_position = parent_position  # to trace back path
        self.holds = holds  # set of 4 IDs (hands and feet)
        self.reachable_holds_left_hand = reachable_holds_left_hand  # set of hold IDs
        self.reachable_holds_right_hand = reachable_holds_right_hand  # set of hold IDs
        self.reachable_holds_left_foot = reachable_holds_left_foot  # set of hold IDs
        self.reachable_holds_right_foot = reachable_holds_right_foot  # set of hold IDs
        self.reachable_positions = reachable_positions  # set of Position IDs

        # booleans to determine if position is start or end position
        self.start = start
        self.end = end

        # tuples to store limb coordinates
        self.left_hand = left_hand
        self.left_elbow = left_elbow
        self.left_shoulder = left_shoulder
        self.left_hip = left_hip
        self.left_knee = left_knee
        self.left_foot = left_foot
        self.right_hand = right_hand
        self.right_elbow = right_elbow
        self.right_shoulder = right_shoulder
        self.right_hip = right_hip
        self.right_knee = right_knee
        self.right_foot = right_foot