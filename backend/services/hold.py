class Hold:
    def __init__(self, wall, location, color, end, yMax, difficulty, difficulty_right, difficulty_top_right, difficulty_top, difficulty_top_left, difficulty_left, difficulty_bottom_left, difficulty_bottom, difficulty_bottom_right, id):
        
        self.wall = wall #this way we enforce a wall object to be connected to a hold
        self.location = location  # should be a 2D array
        self.color = color
        self.end = end
        self.yMax = yMax # should be a 2D array #TODO: use yMax as HoldID everywhere
        self.difficulty = difficulty
        self.difficulty_right = difficulty_right
        self.difficulty_top_right = difficulty_top_right
        self.difficulty_top = difficulty_top
        self.difficulty_top_left = difficulty_top_left
        self.difficulty_left = difficulty_left
        self.difficulty_bottom_left = difficulty_bottom_left
        self.difficulty_bottom = difficulty_bottom
        self.difficulty_bottom_right = difficulty_bottom_right

        self.id = id


