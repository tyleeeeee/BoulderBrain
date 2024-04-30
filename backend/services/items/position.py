class Position:
    def __init__(self, id, timestamp, climber_id, parent_position_id, holds, reachable_holds_left_hand, reachable_holds_right_hand, reachable_holds_left_foot, reachable_holds_right_foot, reachable_positions, start, end, left_hand, left_elbow, left_shoulder, left_hip, left_knee, left_foot, right_hand, right_elbow, right_shoulder, right_hip, right_knee, right_foot):
        self.id = id
        self.timestamp = timestamp  # depth in search tree
        self.climber_id = climber_id
        self.parent_position_id = parent_position_id  # to trace back path
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