import subprocess
import os
import pandas as pd
import numpy as np

from utils import modify_file_by_lines
from constants import EARTH_ATM, ABSOLUTE_ZERO

PHREEQC_path = 'external/phreeqc/bin'

output_file_path = f'{PHREEQC_path}/output.txt'

def find_partial_pressure(P: float, T: float, carbon_molality: float, pH: float | None =None) -> tuple[float, float]:

    input_template_file_path = 'templates/partial_pressure_input.txt'
    input_file_path_new = f'{PHREEQC_path}/input'
    wd: str = os.getcwd()

    input_modifications: dict[int, str] = {
        1 : f'DATABASE {wd}/external/phreeqc-3.8.6-17100/database/phreeqc.dat',
        4 : f'    temp        {T + ABSOLUTE_ZERO:.4f}        # Temperature in degrees Celsius',
        5 : f'    pressure    {P / EARTH_ATM:.4f}         # Pressure in atmospheres',
        8 : f'    C           {carbon_molality:.8f}         # Total dissolved carbon'
    }

    if pH is not None:
        input_modifications[6] = f'    pH          {pH:.2f}'

    modify_file_by_lines(input_template_file_path, input_file_path_new, input_modifications)

    subprocess.run(['./phreeqc', 'input'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=PHREEQC_path)

    solution_df: pd.DataFrame = pd.read_table(output_file_path, sep='\s+') # type: ignore

    si_CO2 = float(solution_df.at[0, 'si_CO2(g)']) # type: ignore
    si_H2O = float(solution_df.at[0, 'si_H2O(g)']) # type: ignore

    P_CO2 = (10 ** si_CO2) * EARTH_ATM
    P_H2O = (10 ** si_H2O) * EARTH_ATM

    return P_CO2, P_H2O

def reverse_partial_pressure(P: float, T: float, P_CO2: float):
    pass

def find_carbon_molality(P: float, T: float, P_CO2: float):

    input_template_file_path = 'templates/molality_input.txt'
    input_file_path_new = f'{PHREEQC_path}/input'
    wd: str = os.getcwd()

    input_modifications: dict[int, str] = {
        1 : f'DATABASE {wd}/external/phreeqc-3.8.6-17100/database/phreeqc.dat',
        4 : f'    temp        {T + ABSOLUTE_ZERO:.4f}        # Temperature in degrees Celsius',
        5 : f'    pressure    {P / EARTH_ATM:.4f}         # Pressure in atmospheres',
        13 : f'    CO2(g) {np.log10(P_CO2 / EARTH_ATM):.4f}'
    }

    modify_file_by_lines(input_template_file_path, input_file_path_new, input_modifications)

    subprocess.run(['./phreeqc', 'input'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=PHREEQC_path)

    solution_df: pd.DataFrame = pd.read_table(output_file_path, sep='\s+') # type: ignore

    molality_C = float(solution_df.at[1, 'C']) # type: ignore

    return molality_C

if __name__ == '__main__':

    P_CO2_original = 0.004 * EARTH_ATM
    molality_original = 0.00206

    P_CO2, P_H2O = find_partial_pressure(EARTH_ATM, 290, molality_original)

    print(f'{P_CO2 / EARTH_ATM:.2%}')
    print(f'{P_H2O / EARTH_ATM:.2%}')



