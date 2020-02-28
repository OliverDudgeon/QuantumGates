import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial

laser_freq = 1
detuning_1 = 0
detuning_2 = 0


def f(laser_freq, detuning_1, detuning_2, time, state):
    return [
        -1j * (laser_freq * (state[1] + state[2])),
        -1j * (detuning_2 * state[1] + laser_freq * (state[3] + state[0])),
        -1j * (detuning_1 * state[2] + laser_freq * (state[3] + state[0])),
        -1j * ((detuning_1 + detuning_2) *
               state[3] + laser_freq * (state[2] + state[2]))
    ]


d = partial(f, laser_freq, detuning_1, detuning_2)

sol = solve_ivp(d, [0, 10], [1+0j, 0+0j, 0+0j, 0+0j], max_step=.01)
print(sol.y)
plt.plot(sol.t, np.abs(sol.y[0])**2)
plt.plot(sol.t, np.abs(sol.y[1])**2)
plt.plot(sol.t, np.abs(sol.y[2])**2)
plt.plot(sol.t, np.abs(sol.y[3])**2)

# Normalisation
norm = np.abs(sol.y[0])**2 + np.abs(sol.y[1])**2 + \
    np.abs(sol.y[2])**2 + np.abs(sol.y[3])**2
plt.plot(sol.t, norm)

plt.show()
