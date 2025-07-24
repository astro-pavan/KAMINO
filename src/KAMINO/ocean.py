default_seawater_ratios = 'input/default_seawater_composition'

import pandas as pd

from external.phreeqc import find_partial_pressures, reverse_partial_pressure, seafloor_equilbrium
from constants import ABSOLUTE_ZERO

class ocean:
    
    def __init__(self, gravity: float, area: float, depth: float, P_surface: float, T_surface: float, salinity: float):
        
        self.gravity = gravity
        self.area = area
        self.depth = depth
        self.P_surface = P_surface
        self.T_surface = T_surface

        self.salinity = salinity # in mol of salt per kg water

        self.composition: dict[str, float] = {} # in mol / kg

        input_composition: pd.DataFrame = pd.read_table(default_seawater_ratios, sep='\s+') # type: ignore

        ratio_total: float = 0

        for i, row in input_composition.iterrows(): # type: ignore
            self.composition[row['Element']] = row['Ratio'] # read as mol ratio
            ratio_total += row['Ratio'] # type: ignore
        for k in self.composition.keys():
            self.composition[k] *= salinity / ratio_total

        self.pH: float
        self.alkalinity: float
        self.carbon_molality: float

    def setup(self, P_CO2: float, carbon_molality: float):

        self.carbon_molality = carbon_molality
        self.pH = reverse_partial_pressure(self.P_surface, self.T_surface, P_CO2, self.composition, carbon_molality)
        self.alkalinity = reverse_partial_pressure(self.P_surface, self.T_surface, P_CO2, self.composition, carbon_molality, find_alkalinity=True)

    def get_partial_pressures(self):
        return find_partial_pressures(self.P_surface, self.T_surface, self.composition, self.alkalinity, self.carbon_molality)
    
    def seafloor_weathering(self):

        seawater_density = 1000
        P_seafloor = seawater_density * self.gravity * self.depth
        T_seafloor = 3 + ABSOLUTE_ZERO

        seafloor_equilbrium(P_seafloor, T_seafloor, self.composition, self.alkalinity, self.carbon_molality)


if __name__ == '__main__':
    o1 = ocean(10, 100, 100, 1e5, 280, 0.03)
    o1.setup((400 * 1e-6) * 1e5, 0.001)
    print(o1.alkalinity)
    print(o1.pH)
    P_CO2, P_H20 = o1.get_partial_pressures()
    print(100 * P_H20 / 1e5 )