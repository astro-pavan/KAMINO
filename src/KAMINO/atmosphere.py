import numpy as np

from constants import SPECIES_MMW

class atmosphere:
    
    def __init__(self, gravity: float, area: float, P_surface: float, x_gas: dict[str, float]):
        
        self.gravity = gravity
        self.area = area
        self.x_gas = x_gas
        self.P_surface = P_surface

        self.P_gas: dict[str, float] = {}
        self.calculate_P_gas()
        
        self.mmw = 0
        self.calulate_mmw()

        self.mass = (self.P_surface * self.area) / self.gravity
        self.moles = self.mass / self.mmw

        self.n_gas: dict[str, float] = {}
        self.calculate_n_gas()

        n_layers = 30

        self.P = np.empty(n_layers, dtype=np.float64)
        self.T = np.empty(n_layers, dtype=np.float64)
        self.z = np.empty(n_layers, dtype=np.float64)

    def calculate_P_gas(self):
        for species in self.P_gas.keys():
            self.P_gas[species] = self.x_gas[species] * self.P_surface

    def calculate_n_gas(self):
        for species in self.x_gas.keys():
            self.n_gas[species] = self.x_gas[species] * self.moles

    def calulate_mmw(self):
        self.mmw = 0
        for species in self.x_gas.keys():
            self.mmw += self.x_gas[species] * SPECIES_MMW[species]

    def add_species(self, amount: float, added_species: str):

        self.n_gas[added_species] += amount
        self.moles += amount
        self.P_surface += (amount * SPECIES_MMW[added_species] * self.gravity) / self.area
        
        for species in self.x_gas.keys():
            self.x_gas[species] = self.n_gas[species] / self.moles

        self.calulate_mmw()
        self.calculate_P_gas()

    def set_partial_pressure(self, new_partial_pressure: float, modified_species: str):
        
        delta_P = new_partial_pressure - self.P_surface * self.x_gas[modified_species]
        self.P_surface += delta_P
        self.P_gas[modified_species] = new_partial_pressure

        for species in self.x_gas.keys():
            self.x_gas[species] = self.P_gas[species] / self.P_surface
        self.calulate_mmw()

        delta_n = (delta_P * self.area) / (SPECIES_MMW[modified_species] * self.gravity)
        self.moles += delta_n
        
        self.calculate_n_gas()

    def radiative_convective_equilbrium(self):
        pass