import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial

from utils.formatter import multiple_formatter

def constrain_phase(phase, t, atol=1.1):
    """
    Get a smooth curve of the phase by adding 2pi at the discontinuities
    """
    # Locate the sign changes as these are where the discontinuities occur
    sign = np.where(np.diff(np.sign(phase)) != 0)[0] + 1
    # Remove the roots where it smoothly crosses through 0
    log = np.logical_not(np.isclose(phase[sign], 0, atol=atol))

    # Apply the shifts at each discontinuity
    for p in t[sign-1][log]:
        phase = np.where(t > p, phase + 2*np.pi, phase)

    return phase


def sine_squared(amplitude, characteristic_time, time):
    return amplitude * np.sin(time*np.pi/characteristic_time)**2


def cos_squared(amplitude, characteristic_time, time):
    return amplitude * np.cos(time*np.pi/characteristic_time)**2


def f(laser_func, detuning_func_1, detuning_func_2, V, time, state):
    # Factor of 0.5 due to rotating wave approximation
    laser_freq = 0.5 * laser_func(time)
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


def solve_with(*, laser=None, detuning_1=None, detuning_2=None, detuning=None, V=1.7, T=1148.5, init=None):

    laser_func = func_make(laser, default_laser_func)
    if detuning is None:
        detuning_1_func = func_make(detuning_1, default_detuning)
        detuning_2_func = func_make(detuning_2, default_detuning)
    else:
        detuning_1_func = detuning_2_func = func_make(detuning, default_detuning)

    if init is None:
        init = [1+0j, 0+0j, 0+0j, 0+0j]

    d = partial(f, laser_func, detuning_1_func, detuning_2_func, V)

    return solve_ivp(d, [0, T], init, t_eval=np.linspace(0, T, 1000))


if __name__ == '__main__':
    plt.style.use('seaborn-talk')

    # Need both simulations in order to
    sol1 = solve_with(V=0)
    sol2 = solve_with(V=1.7)

    # Individual phases
    c11_noint = constrain_phase(np.angle(sol1.y[0]), sol1.t)
    c11_int = constrain_phase(np.angle(sol2.y[0]), sol2.t)

    entangled_phase = c11_noint - c11_int

    plt.plot(sol1.t, c11_noint, label='$c_{11}(V=0)$')
    plt.plot(sol1.t, c11_int, label='$c_{11}(V=1.7)$')
    plt.plot(sol1.t, entangled_phase,
             label=f'Entangled Phase, Phase Change = {(entangled_phase[-1] - entangled_phase[0])/np.pi:.3f}')

    # Plot y-axis in terms of multiples of pi
    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    plt.xlabel('Time, $t$')
    plt.ylabel('Phase')
    plt.legend(frameon=False, ncol=5, loc='upper center',
               bbox_to_anchor=(0.5, 1.1))
    plt.savefig('twoatom_numerical.svg', bbox_inches='tight')

    plt.show()
