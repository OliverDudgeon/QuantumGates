import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial


def f(laser_func, detuning_1, detuning_2, V, time, state):
    laser_freq = laser_func(time)
    return [
        -1j * (laser_freq * (state[1] + state[2])),
        -1j * (detuning_2 * state[1] + laser_freq * (state[3] + state[0])),
        -1j * (detuning_1 * state[2] + laser_freq * (state[3] + state[0])),
        -1j * ((detuning_1 + detuning_2 + V) *
               state[3] + laser_freq * (state[1] + state[2]))
    ]


def default_laser_func(t):
    return 1


def solve_with(*, laser=None, detuning_1=0.5, detuning_2=0.5, V=1, tf=10, init=None):
    if laser is None:
        laser_func = default_laser_func
    elif type(laser) in (float, int, complex):
        def laser_func(t): return laser
    else:
        laser_func = laser
    if init is None:
        init = [1+0j, 0+0j, 0+0j, 0+0j]
    d = partial(f, laser_func, detuning_1, detuning_2, V)

    return solve_ivp(d, [0, tf], init, max_step=.01)


def sine_squared(amplitude, characteristic_time, time):
    return amplitude * np.sin(time/characteristic_time)**2


if __name__ == '__main__':

    sol = solve_with(laser=partial(sine_squared, 2, 5))

    # Individual Prob. Amps
    plt.plot(sol.t, np.abs(sol.y[0])**2, label='$|c_{11}|^2$')
    plt.plot(sol.t, np.abs(sol.y[1])**2, label='$|c_{1r}|^2$')
    plt.plot(sol.t, np.abs(sol.y[2])**2, label='$|c_{r1}|^2$', linestyle='--')
    plt.plot(sol.t, np.abs(sol.y[3])**2, label='$|c_{rr}|^2$')

    norm = (np.abs(sol.y[0])**2
            + np.abs(sol.y[2])**2
            + np.abs(sol.y[3])**2
            + np.abs(sol.y[1])**2)
    plt.plot(sol.t, norm, label='Normalisation')

    plt.xlabel('Time, $t$')
    plt.ylabel('Probability Amplitude')
    plt.legend(frameon=False, ncol=5, loc='upper center',
               bbox_to_anchor=(0.5, 1.1))
    plt.savefig('twoatom_numerical.png', dpi=100)

    plt.show()
