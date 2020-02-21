import numpy as np
import matplotlib.pyplot as plt


laser_freq = 1
detuning = .4

pm = np.array([-1, 1])
eigenvalues = 0.5 * (detuning + pm * np.sqrt(detuning**2 + 4*laser_freq**2))

eigenvectors = np.array([[1, 1], eigenvalues/laser_freq]).T

# print(eigenvectors)
# print(np.abs(eigenvectors)**2)
# print(np.sum(np.abs(eigenvectors)**2, axis=1))
eigenvectors = eigenvectors / np.array([np.sum(np.abs(eigenvectors)**2, axis=1)]).T

alpha = np.sqrt(1 + (eigenvalues/laser_freq)**2)

k1 = alpha[0]*eigenvalues[1]/(eigenvalues[1]-eigenvalues[0])
k2 = alpha[1]*eigenvalues[0]/(eigenvalues[0]-eigenvalues[1])

t = np.array([np.linspace(0, 10, 1_000)]).T

cs = k1 * np.exp(-1j * eigenvalues[0] * t) * np.array([eigenvectors[0]]) + \
    k2 * np.exp(-1j * eigenvalues[1] * t) * np.array([eigenvectors[1]])
# print(cs)

plt.plot(t, np.abs(cs)**2)
plt.plot(t, np.sum(np.abs(cs)**2, axis=1))

plt.show()
