class Route:
    def __init__(self, wall, climber, holds, starting_holds, ending_hold, length, difficulty):
        self.wall = wall  # Wall object
        self.climber = climber # climber object
        self.holds = holds
        self.starting_holds = starting_holds
        self.ending_hold = ending_hold
        self.length = length
        self.difficulty = difficulty
