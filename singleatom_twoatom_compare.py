import matplotlib.pyplot as plt
import numpy as np

from singleatom_numerical import solve_with as solve_singleatom
from twoatom_numerical import solve_with as solve_twoatom

sol1 = solve_singleatom(laser_freq=1, detuning=0.5, tf=10, init=[1+0j, 0+0j])
sol2 = solve_twoatom(laser=2, detuning=0.5, T=10, init=[
                     1+0j, 0+0j, 0+0j, 0+0j], V=0)

fig, axs = plt.subplots(2, sharex=True, sharey=True)  # Creates a figure window

axs[0].plot(sol1.t, np.abs(sol1.y[0])**2, label='$|c_1|^2$')
axs[0].plot(sol1.t, np.abs(sol1.y[1])**2, label='$|c_r|^2$')

# Combined Prob. Amps
axs[1].plot(sol2.t, np.abs(sol2.y[0])**2 + np.abs(sol2.y[2]) ** 2,
            label='$|c_{11}|^2 + |c_{r1}|^2$', ls='--')
axs[1].plot(sol2.t, np.abs(sol2.y[1])**2 + np.abs(sol2.y[3]) ** 2,
            label='$|c_{1r}|^2 + |c_{rr}|^2$', ls='--')

axs[0].set(xlabel='Time, $t$', ylabel='Probability')
axs[1].set(xlabel='Time, $t$', ylabel='Probability')

axs[0].legend(frameon='false', ncol=5, loc='upper right')
axs[1].legend(frameon='false', ncol=5, loc='upper right')
plt.savefig('single_twoatom_compare.pdf')
plt.show()
