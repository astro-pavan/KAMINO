import subprocess
import os
import pandas as pd
import numpy as np
from scipy.optimize import root_scalar # type: ignore

from typing import Union

from utils import modify_file_by_lines, insert_lines_into_file
from constants import EARTH_ATM, ABSOLUTE_ZERO

PHREEQC_path = 'external/phreeqc/bin'

output_file_path = f'{PHREEQC_path}/output.txt'

def find_partial_pressures(P: float, T: float, composition: dict[str, float], alkalinity: Union[float, None]=None, carbon_molality: Union[float, None] = None, pH: Union[float, None]=None) -> tuple[float, float]:

    input_template_file_path = 'input/partial_pressure_input.txt'
    input_file_path_new = f'{PHREEQC_path}/input'
    wd: str = os.getcwd()

    input_modifications: dict[int, str] = {
        1 : f'DATABASE {wd}/external/phreeqc-3.8.6-17100/database/phreeqc.dat',
        4 : f'    temp        {T + ABSOLUTE_ZERO:.4f}        # Temperature in degrees Celsius',
        5 : f'    pressure    {P / EARTH_ATM:.4f}         # Pressure in atmospheres',
    }

    modify_file_by_lines(input_template_file_path, input_file_path_new, input_modifications)

    input_newlines: list[str] = []

    if pH is not None:
        input_newlines.append(f'    pH    {pH:.2f}')
    if carbon_molality is not None:
        input_newlines.append(f'    C    {carbon_molality}')
    if alkalinity is not None:
        input_newlines.append(f'    Alkalinity    {alkalinity:.8f} as HCO3')

    for k in composition.keys():
        input_newlines.append(f'    {k}    {composition[k]:.8f}')

    insert_lines_into_file(input_file_path_new, input_newlines, 6)

    subprocess.run(['./phreeqc', 'input'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=PHREEQC_path)

    solution_df: pd.DataFrame = pd.read_table(output_file_path, sep='\s+') # type: ignore

    si_CO2 = float(solution_df.at[0, 'si_CO2(g)']) # type: ignore
    si_H2O = float(solution_df.at[0, 'si_H2O(g)']) # type: ignore

    P_CO2 = (10 ** si_CO2) * EARTH_ATM
    P_H2O = (10 ** si_H2O) * EARTH_ATM

    return (P_CO2, P_H2O)

def reverse_partial_pressure(P: float, T: float, P_CO2: float, composition: dict[str, float], carbon_molality: float, find_alkalinity: bool=False) -> float:
    
    def P_CO2_function_alk(alkalinity: float): 
        return find_partial_pressures(P, T, composition, alkalinity, carbon_molality)[0] - P_CO2
    
    def P_CO2_function_pH(pH: float): 
        return find_partial_pressures(P, T, composition, pH=pH, carbon_molality=carbon_molality)[0] - P_CO2

    if find_alkalinity:
        result = root_scalar(P_CO2_function_alk, bracket=(0, 55), method='brentq')  # type: ignore
    else:
        result = root_scalar(P_CO2_function_pH, bracket=(0, 14), method='brentq')  # type: ignore

    if result.converged:  # type: ignore
        return result.root  # type: ignore
    else:
        raise ValueError("Root finding did not converge")
    
def seafloor_equilbrium(P_seafloor:float, T_seafloor:float, composition: dict[str, float], alkalinity: Union[float, None]=None, carbon_molality: Union[float, None] = None, pH: Union[float, None]=None) -> tuple[dict[str, float], float, float]:
    
    input_template_file_path = 'input/seafloor_weathering_input.txt'
    input_file_path_new = f'{PHREEQC_path}/input'
    wd: str = os.getcwd()

    input_modifications: dict[int, str] = {
        1 : f'DATABASE {wd}/external/phreeqc-3.8.6-17100/database/phreeqc.dat',
        12: f'    {P_seafloor / EARTH_ATM:.4f}',
        14: f'    {T_seafloor + ABSOLUTE_ZERO:.4f}'
    }

    modify_file_by_lines(input_template_file_path, input_file_path_new, input_modifications)

    input_newlines: list[str] = []

    if pH is not None:
        input_newlines.append(f'    pH    {pH:.2f}')
    if carbon_molality is not None:
        input_newlines.append(f'    C    {carbon_molality}')
    if alkalinity is not None:
        input_newlines.append(f'    Alkalinity    {alkalinity:.8f} as HCO3')

    for k in composition.keys():
        input_newlines.append(f'    {k}    {composition[k]:.8f}')

    insert_lines_into_file(input_file_path_new, input_newlines, 5)

    subprocess.run(['./phreeqc', 'input'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=PHREEQC_path)

    solution_df: pd.DataFrame = pd.read_table(output_file_path, sep='\s+') # type: ignore

    # print(solution_df.keys())

    new_composition: dict[str, float]= {}

    for k in composition:
        new_composition[k] = float(np.array(solution_df[k])[1]) # type: ignore

    new_alkalinity = float(np.array(solution_df['Alkalinity'])[1]) # type: ignore
    new_carbon_molality = float(np.array(solution_df['C'])[1]) # type: ignore

    return new_composition, new_alkalinity, new_carbon_molality

if __name__ == '__main__':

    sal = 1

    comp: dict[str, float] = {
        'Cl': 0.546 * sal,
        'Na': 0.469 * sal,
        'Mg': 0.0528 * sal,
        'S(6)': 0.0282 * sal,
        'Ca': 0.0103 * sal,
        'K': 0.0102 * sal
    }

    # comp: dict[str, float] = {
    #     'Cl': 0.546,
    #     'Na': 0.469,
    #     'Mg': 0.0528,
    #     'S(6)': 0.0282,
    #     'Ca': 0.0103,
    #     'K': 0.0102
    # }

    alk = 1000 * 1e-6
    C = 0.001

    dT = +3

    T_bottom = 277
    T_bottom_new = T_bottom + dT

    # T_bottom = 273 + 400
    # T_bottom_new = 273 + 403

    T_0 = 293
    T_0_new = T_0 + dT

    P_floor = 500

    print(f'Alkanlinity : {alk}, C : {C}')

    new_comp, new_alk, new_C = seafloor_equilbrium(P_floor * EARTH_ATM, T_bottom, comp, alkalinity=alk, carbon_molality=C)
    P_CO2_old, P_H2O = find_partial_pressures(1 * EARTH_ATM, T_0, new_comp, new_alk, new_C)

    print(f'Alkanlinity : {new_alk}, C : {new_C}, P_CO2: {P_CO2_old / EARTH_ATM:.4%}, P_H2O: {P_H2O / EARTH_ATM:.4%}')

    new_comp, new_alk, new_C = seafloor_equilbrium(P_floor * EARTH_ATM, T_bottom, new_comp, alkalinity=new_alk, carbon_molality=new_C)
    P_CO2, P_H2O = find_partial_pressures(1 * EARTH_ATM, T_0_new, new_comp, new_alk, new_C) # type: ignore

    print(f'Alkanlinity : {new_alk}, C : {new_C}, P_CO2: {P_CO2 / EARTH_ATM:.4%}, P_H2O: {P_H2O / EARTH_ATM:.4%}')

    new_comp, new_alk, new_C = seafloor_equilbrium(P_floor * EARTH_ATM, T_bottom_new, new_comp, alkalinity=new_alk, carbon_molality=new_C)
    P_CO2, P_H2O = find_partial_pressures(1 * EARTH_ATM, T_0_new, new_comp, new_alk, new_C) # type: ignore

    print(f'!! Alkanlinity : {new_alk}, C : {new_C}, P_CO2: {P_CO2 / EARTH_ATM:.4%}, P_H2O: {P_H2O / EARTH_ATM:.4%} !!')

    new_comp, new_alk, new_C = seafloor_equilbrium(P_floor * EARTH_ATM, T_bottom_new, new_comp, alkalinity=new_alk, carbon_molality=new_C)
    P_CO2, P_H2O = find_partial_pressures(1 * EARTH_ATM, T_0, new_comp, new_alk, new_C) # type: ignore

    print(f'Alkanlinity : {new_alk}, C : {new_C}, P_CO2: {P_CO2 / EARTH_ATM:.4%}, P_H2O: {P_H2O / EARTH_ATM:.4%}')

    # P_CO2, P_H2O = find_partial_pressures(EARTH_ATM, 290, comp, alkalinity=2223 * 1e-6, carbon_molality=0.002)

    # print(f'{P_CO2 / EARTH_ATM:.4%}')
    # print(f'{P_H2O / EARTH_ATM:.2%}')

    # alk = reverse_partial_pressure(EARTH_ATM, 290, (400 * 1e-6) * EARTH_ATM, comp, carbon_molality=0.002, find_alkalinity=True)

    # print(alk * 1e6)

