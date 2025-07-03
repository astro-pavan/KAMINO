class ocean:
    
    def __init__(self, gravity: float, area: float, depth: float, P_surface: float, T_surface: float):
        
        self.gravity = gravity
        self.area = area
        self.depth = depth
        self.P_surface = P_surface
        self.T_surface = T_surface

        self.molality: dict[str, float] = {}
        self.n_species: dict[str, float] = {}

    def add_species(self, amount: float, added_species: str):
        pass