import numpy as np
import matplotlib.pyplot as plt


def solve_with(*, laser_freq=1, detuning=.0):
    pm = np.array([-1, 1])
    eigenvalues = 0.5 * (detuning
                         + pm*np.sqrt(detuning**2 + 4*laser_freq**2))

    eigenvectors = np.array([[1, 1], eigenvalues/laser_freq]).T

    eigenvectors = (eigenvectors
                    / np.array([np.sum(np.abs(eigenvectors)**2, axis=1)]).T)

    k1 = 1
    k2 = 1

    t = np.array([np.linspace(0, 10, 1_000)]).T

    return t, (k1*np.exp(-1j*eigenvalues[0] * t) * np.array([eigenvectors[0]])
               + k2*np.exp(-1j*eigenvalues[1] * t) * np.array([eigenvectors[1]]))


if __name__ == '__main__':
    t, cs = solve_with()
    plt.plot(t, np.abs(cs[:, 0])**2, label='$|c_1|^2$')
    plt.plot(t, np.abs(cs[:, 1])**2, label='$|c_r|^2$')
    plt.plot(t, np.sum(np.abs(cs)**2, axis=1), label='normalisation')

    plt.xlabel('Time, $t$')
    plt.ylabel('Probability Amplitude')
    plt.legend(frameon=False, ncol=5, loc='upper center',
               bbox_to_anchor=(0.5, 1.1))
    plt.savefig('single_atom_analytical.png', dpi=100)
    plt.show()
