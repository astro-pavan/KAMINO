# type: ignore

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

h_b = 200
a_0 = 50

def K(z: float, h: float=h_b) -> float:
    # res = float(np.where(z < 0.8 * h, 1e-2, 1e-4))
    # res = float(np.where((z < 1.2 * h) & (z > 0.8 * h), 1e-5, 1e-4))

    if z < 0.9 * h_b:
        Kz = 1e-1
    elif 0.9 * h_b < z < 1.2 * h_b:
        Kz = 1e-5
    else:
        Kz = 0.7e-5

    return Kz

def dTdz(z: float, I0: float, a: float, rho: float=1000, cp: float=4184) -> float:
    return - (1 / K(z)) * (I0 / (rho * cp)) * np.exp(-z / a)

if __name__ == '__main__':

    f = lambda z, T: dTdz(z, 272, a_0)
   
    T0 = 15
    sol = integrate.solve_ivp(f, (0, 3000), [T0], t_eval=np.linspace(0, 3000, num=300))
    
    T1 = sol.y[0]
    z1 = sol.t

    T0 = 16
    sol = integrate.solve_ivp(f, (0, 3000), [T0], t_eval=np.linspace(0, 3000, num=300))
    
    T2 = sol.y[0]
    z2 = sol.t

    T0 = 17
    sol = integrate.solve_ivp(f, (0, 3000), [T0], t_eval=np.linspace(0, 3000, num=300))
    
    T3 = sol.y[0]
    z3 = sol.t

    T0 = 14
    sol = integrate.solve_ivp(f, (0, 3000), [T0], t_eval=np.linspace(0, 3000, num=300))
    
    T4 = sol.y[0]
    z4 = sol.t

    T0 = 13
    sol = integrate.solve_ivp(f, (0, 3000), [T0], t_eval=np.linspace(0, 3000, num=300))
    
    T5 = sol.y[0]
    z5 = sol.t

    plt.plot(T1, z1)
    plt.plot(T2, z2)
    plt.plot(T3, z3)
    plt.plot(T4, z4)
    plt.plot(T5, z5)
    plt.axhline(h_b)
    plt.axhline(a_0)
    plt.ylim([3000, 0])
    plt.xlim([0, 20])
    plt.savefig('profile.png')

