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
    
def seafloor_equilbrium(P_seafloor:float, T_seafloor:float, composition: dict[str, float], minerals: list[str], alkalinity: Union[float, None]=None, carbon_molality: Union[float, None] = None, pH: Union[float, None]=None) -> tuple[dict[str, float], float, float]:
    
    input_template_file_path = 'input/seafloor_weathering_input.txt'
    input_file_path_new = f'{PHREEQC_path}/input'
    wd: str = os.getcwd()

    mineral_list = ''
    for mineral in minerals:
        mineral_list += mineral + ' '

    input_modifications: dict[int, str] = {
        1 : f'DATABASE {wd}/external/phreeqc-3.8.6-17100/database/phreeqc.dat',
        9: f'    {P_seafloor / EARTH_ATM:.4f}',
        11: f'    {T_seafloor + ABSOLUTE_ZERO:.4f}',
        16: f'    -equilibrium_phases {mineral_list}',
        17: f'    -saturation_indices {mineral_list}'
    }

    modify_file_by_lines(input_template_file_path, input_file_path_new, input_modifications)

    input_newlines: list[str] = []
    for mineral in minerals:
        input_newlines.append(f'    {mineral}   0.0 10.0')
    insert_lines_into_file(input_file_path_new, input_newlines, 6)

    input_newlines: list[str] = []

    if pH is not None:
        input_newlines.append(f'    pH    {pH:.2f}')
    if carbon_molality is not None:
        input_newlines.append(f'    C    {carbon_molality}')
    if alkalinity is not None:
        input_newlines.append(f'    Alkalinity    {alkalinity:.8f} as HCO3')

    for k in composition.keys():
        input_newlines.append(f'    {k}    {composition[k]:.8f}')

    insert_lines_into_file(input_file_path_new, input_newlines, 4)

    subprocess.run(['./phreeqc', 'input'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=PHREEQC_path)

    solution_df: pd.DataFrame = pd.read_table(output_file_path, sep='\s+') # type: ignore

    # print(solution_df.keys())

    new_composition: dict[str, float]= {}

    for k in composition:
        new_composition[k] = float(np.array(solution_df[k])[1]) # type: ignore

    new_alkalinity = float(np.array(solution_df['Alkalinity'])[1]) # type: ignore
    new_carbon_molality = float(np.array(solution_df['C'])[1]) # type: ignore

    return new_composition, new_alkalinity, new_carbon_molality

def seafloor_equilbrium_v2(P_seafloor:float, T_seafloor:float, composition: dict[str, float], minerals: list[str], alkalinity: Union[float, None]=None, carbon_molality: Union[float, None] = None, pH: Union[float, None]=None) -> tuple[dict[str, float], float, float]:
    
    input_template_file_path = 'input/seafloor_weathering_input_v2.txt'
    input_file_path_new = f'{PHREEQC_path}/input'
    wd: str = os.getcwd()

    mineral_list = ''
    for mineral in minerals:
        mineral_list += mineral + ' '

    #external/phreeqc-3.8.6-17100/database/Kinec_v3.dat

    input_modifications: dict[int, str] = {
        1 : f'DATABASE {wd}/external/phreeqc-3.8.6-17100/database/Kinec_v3.dat',
        6: f'    pressure {P_seafloor / EARTH_ATM:.4f}',
        5: f'    temp {T_seafloor + ABSOLUTE_ZERO:.4f}',
        16: f'    -equilibrium_phases {mineral_list}',
        17: f'    -saturation_indices {mineral_list}'
    }

    modify_file_by_lines(input_template_file_path, input_file_path_new, input_modifications)

    input_newlines: list[str] = []
    for mineral in minerals:
        input_newlines.append(f'    {mineral}   0.0 10.0')
    insert_lines_into_file(input_file_path_new, input_newlines, 8)

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
        'K': 0.0102 * sal,
        'Si': 0.0000 * sal,
        'Al': 0.00000 * sal
    }

    P_surface = 2 * EARTH_ATM
    T_surface = 293
    original_C_molality = 0.002

    # find_partial_pressures(P_surface, 310, comp, carbon_molality=0.1, pH=7)
    seafloor_equilbrium_v2(P_seafloor=10*1e5, T_seafloor=280, composition=comp, minerals=['Calcite', 'Aragonite'], carbon_molality=0.1, pH=7)

    # new_composition, new_alkalinity, new_carbon_molality = seafloor_equilbrium(400 * EARTH_ATM, 275, comp, ['Anorthite'], original_alkalinity, original_C_molality)

    # print(new_composition)
    # print(new_carbon_molality)

    # new_composition, new_alkalinity, new_carbon_molality = seafloor_equilbrium(400 * EARTH_ATM, 275, comp, ['Anorthite'], original_alkalinity, original_C_molality)

    # print(new_composition)
    # print(new_carbon_molality)
