import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from functools import partial


def f(laser_freq, detuning, time, state):
    """
    Calculate derivatives in from coupled differential eqns
    state is composed of [c1, cr]
    """
    return [-1j*laser_freq*state[1], -1j*(laser_freq*state[0] + detuning*state[1])]


def solve_with(*, laser_freq=1, detuning=0.5, tf=10, init=None):
    if init is None:
        init = [1+0j, 0+0j]

    d = partial(f, laser_freq, detuning)

    return solve_ivp(d, [0, tf], init, max_step=.01)


if __name__ == "__main__":
    sol = solve_with()

    # Plotting modulus square of c1, cr
    plt.plot(sol.t, np.abs(sol.y[0])**2, label='$|c_1|^2$')
    plt.plot(sol.t, np.abs(sol.y[1])**2, label='$|c_r|^2$')

    # Normalisation
    plt.plot(sol.t, np.abs(sol.y[0])**2 +
             np.abs(sol.y[1])**2, label='normalisation')

    plt.xlabel('Time, $t$')
    plt.ylabel('Probability Amplitude')
    plt.legend(frameon=False, ncol=5, loc='upper center',
               bbox_to_anchor=(0.5, 1.1))
    plt.savefig('single_atom_numerical.png', dpi=100)
    plt.show()
