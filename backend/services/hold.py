class Hold:
    def __init__(self, wall, location, color, end, yMax):
        self.id = id
        self.wall = wall #this way we enforce a wall object to be connected to a hold
        self.location = location  # should be a 2D array
        self.color = color
        self.end = end
        self.yMax = yMax # should be a 2D array #TODO: do we need location AND yMax?

