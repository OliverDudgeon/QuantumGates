import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial

laser_freq = 1
detuning_1 = 0.5
detuning_2 = 0.5


def f(laser_freq, detuning_1, detuning_2, time, state):
    return [
        -1j * (laser_freq * (state[1] + state[2])),
        -1j * (detuning_2 * state[1] + laser_freq * (state[3] + state[0])),
        -1j * (detuning_1 * state[2] + laser_freq * (state[3] + state[0])),
        -1j * ((detuning_1 + detuning_2) *
               state[3] + laser_freq * (state[1] + state[2]))
    ]


d = partial(f, laser_freq, detuning_1, detuning_2)

sol = solve_ivp(d, [0, 10], [0+0j, 0+0j, 0+0j, 1+0j], max_step=.01)

plt.plot(sol.t, np.abs(sol.y[0])**2, label='$|c_{11}|^2$')
plt.plot(sol.t, np.abs(sol.y[1])**2, label='$|c_{1r}|^2$')
plt.plot(sol.t, np.abs(sol.y[2])**2, label='$|c_{r1}|^2$', linestyle='--')
plt.plot(sol.t, np.abs(sol.y[3])**2, label='$|c_{rr}|^2$')

# Normalisation
norm = np.abs(sol.y[0])**2 + np.abs(sol.y[1])**2 + \
    np.abs(sol.y[2])**2 + np.abs(sol.y[3])**2
plt.plot(sol.t, norm, label='Normalisation')

plt.xlabel('Time, $t$')
plt.ylabel('Probability Amplitude')
plt.legend()
plt.show()

