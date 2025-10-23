import numpy as np

from constants import G

class planet:

    def __init__(self, R: float, M: float, ocean_depth: float, P_surface: float):
        
        n_atm_levels = 30
        n_ocean_levels = 100

        # PLANET PROPERTIES

        self.R_planet = R
        self.M_planet = M
        self.g = (G * M) / (R ** 2)

        self.P_surface = P_surface
        self.T_surface = 0

        # ATMOSPHERE PROPERTIES

        self.z_atm = np.empty(n_atm_levels)
        self.P_atm = np.empty(n_atm_levels)
        self.T_atm = np.empty(n_atm_levels)

        self.P_CO2 = 0
        self.P_H2O = 0

        # OCEAN PROPERTIES

        self.ocean_depth = ocean_depth
        self.V_ocean = 4 * np.pi * ocean_depth * R ** 2
        self.M_ocean = self.V_ocean * 1000

        self.z_ocean = np.linspace(0, -ocean_depth, n_ocean_levels)
        self.P_ocean = P_surface + 1000 * self.g * self.z_ocean
        self.P_seafloor = self.P_ocean[-1]

        self.molality = {}

    def update_atmosphere(self):

        # run PHREEQC here

        self.P_CO2 = 0 # update with PHREEQC
        self.P_H2O = 0

        # run HELIOS here

        self.z_atm = np.empty(10) # update with HELIOS
        self.P_atm = np.empty(10)
        self.T_atm = np.empty(10)

        self.T_surface = self.T_atm[-1]

    def update_ocean_chemistry(self):

        pass

