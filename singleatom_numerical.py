import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial

laser_freq = 1
detuning = .4


def f(laser_freq, detuning, time, state):
    """
    Calculate derivatives in from coupled differential eqns
    state is composed of [c1, cr]
    """
    return [-1j*laser_freq*state[1], -1j*(laser_freq*state[0] + detuning*state[1])]


d = partial(f, laser_freq, detuning)

sol = solve_ivp(d, [0, 10], [1+0j, 0+0j], max_step=.01)

# Plotting modulus square of c1, cr
plt.plot(sol.t, np.abs(sol.y[0])**2)
plt.plot(sol.t, np.abs(sol.y[1])**2)

# Normalisation
plt.plot(sol.t, np.abs(sol.y[0])**2 + np.abs(sol.y[1])**2)

plt.show()
