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


def solve_with(*, laser=None, detuning_1=None, detuning_2=None, V=1.7, tf=10, init=None):

    laser_func = func_make(laser, default_laser_func)
    detuning_1_func = func_make(detuning_1, default_detuning)
    detuning_2_func = func_make(detuning_2, default_detuning)

    if init is None:
        stphs=(0) #Starting phase
        init = [np.cos(stphs)+(np.sin(stphs)*1j) , 0+0j, 0+0j, 0+0j]

    d = partial(f, laser_func, detuning_1_func, detuning_2_func, V)

    return solve_ivp(d, [0, tf], init, max_step=.1)


if __name__ == '__main__':

    sol = solve_with(tf=1148.5)

    # Individual phases
    fig, axs = plt.subplots(2,2, sharex=True, sharey=True)
    
    h=np.angle(sol.y[0])
    print('Phase shift ',abs(h[0]-h[-1])/(np.pi),'pi')
    
    axs[0,0].plot(sol.t, np.angle(sol.y[0]), label='$arg(c_{11})$')
    axs[0,1].plot(sol.t, np.angle(sol.y[1]), label='$arg(c_{1r})$', linestyle='--')
    axs[1,0].plot(sol.t, np.angle(sol.y[2]), label='$arg(c_{r1})$', linestyle='--')
    axs[1,1].plot(sol.t, np.angle(sol.y[3]), label='$arg(c_{rr})$')

    axs[0,0].set_title('$arg(c_{11})$')
    axs[0,1].set_title('$arg(c_{1r})$')
    axs[1,0].set_title('$arg(c_{r1})$')
    axs[1,1].set_title('$arg(c_{rr})$')
    
    ylb = plt.gca()
    ylb.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ylb.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    ylb.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    for ax in axs.flat:
        ax.set(xlabel='Time , $t$', ylabel='Phase')
    for ax in axs.flat:
        ax.label_outer()
   
    #plt.legend(frameon=False, ncol=5, loc='upper center',
     #          bbox_to_anchor=(0.5, 1.1))
    plt.savefig('twoatom_numerical_phase.png', dpi=100)

    plt.show()
