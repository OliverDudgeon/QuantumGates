import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial


def f(laser_func, detuning_func_1, detuning_func_2, V, time, state):
    laser_freq = laser_func(time)
    detuning_1 = detuning_func_1(time)
    detuning_2 = detuning_func_2(time)
    return [
        -1j * (laser_freq * (state[1] + state[2])),
        -1j * (detuning_2 * state[1] + laser_freq * (state[3] + state[0])),
        -1j * (detuning_1 * state[2] + laser_freq * (state[3] + state[0])),
        -1j * ((detuning_1 + detuning_2 + V) *
               state[3] + laser_freq * (state[1] + state[2]))
    ]


def default_laser_func(t):
    return 1


def default_detuning(t):
    return 0.5


def func_make(w, default):
    if w is None:
        func = default
    elif type(w) in (float, int, complex):
        def func(t): return w
    else:
        func = w
    return func


def solve_with(*, laser=None, detuning_1=None, detuning_2=None, V=1.7, tf=1148.5, init=None):

    laser_func = func_make(laser, default_laser_func)
    detuning_1_func = func_make(detuning_1, default_detuning)
    detuning_2_func = func_make(detuning_2, default_detuning)

    if init is None:
        init = [1+0j, 0+0j, 0+0j, 0+0j]

    d = partial(f, laser_func, detuning_1_func, detuning_2_func, V)

    return solve_ivp(d, [0, tf], init, max_step=.1)


def sine_squared(amplitude, characteristic_time, time):
    return amplitude * np.sin((time*np.pi)/characteristic_time)**2


def cos_squared(amplitude, characteristic_time, time):
    return amplitude * np.cos((time*np.pi)/characteristic_time)**2


if __name__ == '__main__':

    sol = solve_with(laser=partial(sine_squared, 0.1, 1148.5),
                     detuning_1=partial(cos_squared, 1.8, 1148.5),
                     detuning_2=partial(cos_squared, 1.8, 1148.5))

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
