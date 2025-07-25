# type: ignore

from seafreeze import seafreeze as sf
# NOTE: "The SeaFreeze package allows to compute the thermodynamic and elastic properties of pure water, ice polymorphs (Ih, II, III, V VI and ice VII/ice X) up to 100 GPa and 10,000K and aqueous NaCl solution up to 8GPa and 2,000K."

import numpy as np
from scipy.optimize import root_scalar
from tqdm import tqdm


def EOS(P: float, T: float, salinity: float, property: str) -> float:

    PTm = np.empty((1,), dtype='object')
    PTm[0] = (P * 1e-6, T, salinity)

    out = sf.getProp(PTm, 'NaClaq')
    
    if property == 'S' or property == 'entropy':
        return out.S[0]
    if property == 'rho' or property == 'density':
        return out.rho[0]
    else:
        raise ValueError("Invalid property")
    
def phase(P: float, T: float, salinity: float) -> str:
    
    PTm = np.empty((1,), dtype='object')
    PTm[0] = (P * 1e-6, T, salinity)
    
    out = sf.whichphase(PTm, solute='NaClaq')

    return sf.phasenum2phase(out[0])
    

def make_adiabat(P_start: float, T_start: float, P_end: float, salinity: float=0, num: int=100):
    
    P_arr = np.logspace(np.log10(P_start), np.log10(P_end), num)
    # P_arr = np.linspace(P_start, P_end, num)
    T_arr = np.empty_like(P_arr)
    rho_arr = np.empty_like(P_arr)

    # Initial entropy at starting point
    S0 = EOS(P_start, T_start, salinity, 'entropy')
    T_arr[0] = T_start
    rho_arr[0] = EOS(P_start, T_start, salinity, 'density')

    print(f'Making adiabat at {P_start:.2e} Pa, {T_start:.0f} K...')

    for i in tqdm(range(1, num)):
        def objective(T):
            return EOS(P_arr[i], T, salinity, 'entropy') - S0

        # Use previous T as initial guess
        res = root_scalar(objective, bracket=[T_arr[i-1] - 50, T_arr[i-1] + 50], method='brentq')
        T_arr[i] = res.root

        rho_arr[i] = EOS(P_arr[i], T_arr[i], salinity, 'density')

    return P_arr, T_arr, rho_arr

if __name__ == '__main__':

    print(phase(1e5, 270, 1.0))

    P_adiabat, T_adiabat, rho_adiabat = make_adiabat(1e5, 300, 1e9)

    print(P_adiabat)
    print(T_adiabat)
    print(rho_adiabat)

