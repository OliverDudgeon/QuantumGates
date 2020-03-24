import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial

from utils.formatter import multiple_formatter


def sine_squared(amplitude, characteristic_time, time):
    return amplitude * np.sin(time*np.pi/characteristic_time)**2


def cos_squared(amplitude, characteristic_time, time):
    return amplitude * np.cos(time*np.pi/characteristic_time)**2


def f(laser_func, detuning_func_1, detuning_func_2, V, time, state):
    laser_freq = laser_func(time)
    detuning_1 = detuning_func_1(time)
    detuning_2 = detuning_func_2(time)

    # print(f'{laser_freq:.5}\t{detuning_1:.5}\t{detuning_2:.5}')

    return [
        -1j * (laser_freq * (state[1] + state[2])),
        -1j * (detuning_2 * state[1] + laser_freq * (state[3] + state[0])),
        -1j * (detuning_1 * state[2] + laser_freq * (state[3] + state[0])),
        -1j * ((detuning_1 + detuning_2 + V) *
               state[3] + laser_freq * (state[1] + state[2]))
    ]


def default_laser_func(t):
    return 0.5 * sine_squared(0.1, 1148.5, t)


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


def solve_with(*, laser=None, detuning_1=None, detuning_2=None, V=1.7, tf=10, init=None):

    laser_func = func_make(laser, default_laser_func)
    detuning_1_func = func_make(detuning_1, default_detuning)
    detuning_2_func = func_make(detuning_2, default_detuning)

    if init is None:
        init = [1+0j, 0+0j, 0+0j, 0+0j]

    d = partial(f, laser_func, detuning_1_func, detuning_2_func, V)

    return solve_ivp(d, [0, tf], init, t_eval=np.linspace(0, tf, 10_000))


if __name__ == '__main__':

    sol1 = solve_with(tf=1148.5, V=0)
    sol2 = solve_with(tf=1148.5, V=1.7)

    # Individual phases
    c1r_phase = np.angle(sol1.y[1])
    c11_phase = np.angle(sol2.y[0])

    entangled_phase = c11_phase - 2*c1r_phase

    plt.plot(sol1.t, entangled_phase, label=r'Entangled Phase')

    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    plt.text(0, 0.5, f'phase_change={(entangled_phase[-1] - entangled_phase[0])/np.pi}', fontsize=12, alpha=.8)
    plt.xlabel('Time, $t$')
    plt.ylabel('Phase')
    plt.legend(frameon=False, ncol=5, loc='upper center',
               bbox_to_anchor=(0.5, 1.1))
    plt.savefig('twoatom_numerical.png', dpi=100)

    plt.show()
