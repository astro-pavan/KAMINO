import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from constants import *
from external.phreeqc import *

n_P, n_T = 60, 64

P_seafloor = np.linspace(100, 2000, num=n_P) * EARTH_ATM
#T_seafloor = np.concatenate([np.linspace(1, 20, num=n_T//2), np.logspace(1.5, 3, num=n_T//2)]) - ABSOLUTE_ZERO
T_seafloor = np.concatenate([np.linspace(1, 20, num=n_T//2), np.logspace(1.5, np.log10(1000), num=n_T//2)]) - ABSOLUTE_ZERO

P_grid, T_grid = np.meshgrid(P_seafloor, T_seafloor)

P_surface = 1 * EARTH_ATM
T_surface = 293

salinity = 0

original_comp: dict[str, float] = {
    'Cl': 0.546 * salinity,
    'Na': 0.469 * salinity,
    'Mg': 0.0528 * salinity,
    'S(6)': 0.0282 * salinity,
    'Ca': 0.0103 * salinity,
    'K': 0.0102 * salinity,
    'Si': 0.0000 * salinity,
    'Al': 0.00000 * salinity
}

original_C_molality = 0.002

original_alkalinity = reverse_partial_pressure(P_surface, T_surface, (300 * 1e-6) * EARTH_ATM, original_comp, original_C_molality, find_alkalinity=True)

# print(original_alkalinity)
# print(original_comp)

# original_P_CO2, _ = find_partial_pressures(P_surface, T_surface, original_comp, original_alkalinity, original_C_molality)
# new_P_CO2, _ = find_partial_pressures(P_surface, T_surface + 1, original_comp, original_alkalinity, original_C_molality)


# print(original_P_CO2 - )

def plot_mineral(mineral: str, dT: int=1, name: str='feedback.png'):

    dP_CO2 = np.empty_like(P_grid)

    for i_P in tqdm(range(n_P)):
        for i_T in range(n_T):

            P: float = P_seafloor[i_P]
            T: float = T_seafloor[i_T]

            # equilibrium

            comp, alkalinity, C_molality = seafloor_equilbrium(P, T, original_comp, [mineral], original_alkalinity, original_C_molality)
            # for i in range(12):
            #     comp, alkalinity, C_molality = seafloor_equilbrium(P, T, comp, [mineral], alkalinity, C_molality)

            # print(comp)

            P_CO2_initial, _ = find_partial_pressures(P_surface, T_surface, comp, alkalinity, C_molality)

            new_comp, new_alkalinity, new_C_molality = seafloor_equilbrium(P, T + dT, comp, [mineral], alkalinity, C_molality)

            # print(new_comp)

            P_CO2_new, _ = find_partial_pressures(P_surface, T_surface, new_comp, new_alkalinity, new_C_molality)

            dP_CO2[i_T, i_P] = P_CO2_new - P_CO2_initial

            # print(P_CO2_new - P_CO2_initial)

    vmax = np.max(np.abs(dP_CO2 / 1e5))
    vmin = -vmax

    plt.contourf(T_grid + ABSOLUTE_ZERO, P_grid / 1e5, dP_CO2 / 1e5, 200, cmap='turbo')
    plt.colorbar(label='Change in $P_{CO2}$ (bar)')
    plt.contour(T_grid + ABSOLUTE_ZERO, P_grid / 1e5, dP_CO2 / 1e5, [0])
    plt.xlabel('Seafloor Temperature (C)')
    plt.ylabel('Pressure (bar)')
    plt.xscale('log')
    plt.title(mineral)
    plt.savefig(f'{mineral}_v4.png')
    plt.close()

# Important minerals:
# Carbonates: Calcite, Aragonite
# Plagioclase: Anorthite, Bytownite, Labradorite, Andesine, Oligoclase, Albite
# 

plot_mineral('Calcite')
