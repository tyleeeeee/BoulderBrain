class Wall:
    def __init__(self, id, height, width):
        self.id = id
        self.height = height
        self.width = width
        self.holds = []  # list of Hold objects
        self.routes = []  # list of Route objects

    def add_hold(self, hold):
        self.holds.append(hold)

    def add_route(self, route):
        self.routes.append(route)