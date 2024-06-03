class Hold:
    def __init__(self, wall, location, color, end, yMax, difficulty_1, difficulty_2, difficulty_3, difficulty_4, difficulty_5, difficulty_6, difficulty_7, difficulty_8, id):
        
        self.wall = wall #this way we enforce a wall object to be connected to a hold
        self.location = location  # should be a 2D array
        self.color = color
        self.end = end
        self.yMax = yMax # should be a 2D array #TODO: use yMax as HoldID everywhere
        self.difficulty_1 = difficulty_1
        self.difficulty_2 = difficulty_2
        self.difficulty_3 = difficulty_3
        self.difficulty_4 = difficulty_4
        self.difficulty_5 = difficulty_5
        self.difficulty_6 = difficulty_6
        self.difficulty_7 = difficulty_7
        self.difficulty_8 = difficulty_8

        self.id = id


