import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial

from utils.formatter import multiple_formatter


def sine_squared(amplitude, characteristic_time, time):
    return amplitude * np.sin(time*np.pi/characteristic_time)**2


def cos_squared(amplitude, characteristic_time, time):
    return amplitude * np.cos(time*np.pi/characteristic_time)**2


def f(laser_func, detuning_func, time, state):
    # Factor of 0.5 due to rotating wave approximation
    laser_freq = 0.5 * laser_func(time)
    detuning = detuning_func(time)

    # print(f'{laser_freq:.5}\t{detuning_1:.5}\t{detuning_2:.5}')

    return [-1j*laser_freq*state[1], -1j*(laser_freq*state[0] + detuning*state[1])]


def default_laser_func(t):
    return sine_squared(0.1, 1148.5, t)


def default_detuning(t):
    return cos_squared(1.8, 1148.5, t)


def func_make(w, default):
    if w is None:
        func = default
    elif type(w) in (float, int, complex):
        def func(t): return w
    else:
        func = w
    return func


def solve_with(*, laser=None, detuning_1=None,T=1148.5, init=None):

    laser_func = func_make(laser, default_laser_func)
    detuning_1_func = func_make(detuning_1, default_detuning)

    if init is None:
        init = [1+0j, 0+0j]

    d = partial(f, laser_func, detuning_1_func)

    return solve_ivp(d, [0, T], init, t_eval=np.linspace(0, T, 1000))


if __name__ == '__main__':
    
    sol = solve_with()
    
    phase=np.angle(sol.y[0])
    entangled_phase = np.where(phase > 0.1,phase - 2*np.pi,phase)
    entangled_phase = np.where(phase < -2*np.pi,phase + 2*np.pi,phase)
    # Plotting modulus square of c1, cr
    plt.plot(sol.t,entangled_phase/np.pi)
    #plt.plot(sol.t, np.abs(sol.y[1])**2, label='$|c_r|^2$')

    plt.xlabel('Time, $t$')
    plt.ylabel('Probability Amplitude')

    plt.savefig('single_atom_numerical.png', dpi=100)
    plt.show()
