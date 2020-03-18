import matplotlib.pyplot as plt
import numpy as np

from singleatom_numerical import solve_with as solve_singleatom
from twoatom_numerical import solve_with as solve_twoatom

sol1 = solve_singleatom(laser_freq=1, detuning=0.5, tf=10, init=[1+0j, 0+0j])
sol2 = solve_twoatom(laser=1, detuning_1=0.5,
                     detuning_2=0.5, tf=10, init=[0+0j, 0+0j, 0+0j, 1+0j], V=0)

plt.plot(sol1.t, np.abs(sol1.y[0])**2, label='$|c_1|^2$')
plt.plot(sol1.t, np.abs(sol1.y[1])**2, label='$|c_r|^2$')


# Combined Prob. Amps
plt.plot(sol2.t, np.abs(sol2.y[0])**2 + np.abs(sol2.y[2]) ** 2,
         label='$|c_{11}|^2 + |c_{r1}|^2$', ls='--')
plt.plot(sol2.t, np.abs(sol2.y[1])**2 + np.abs(sol2.y[3]) ** 2,
         label='$|c_{1r}|^2 + |c_{rr}|^2$', ls='--')

plt.xlabel('Time, $t$')
plt.ylabel('Probability Amplitude')
plt.legend(frameon=False, ncol=5, loc='upper center',
           bbox_to_anchor=(0.5, 1.1))
plt.savefig('single_twoatom_compare.png', dpi=100)
plt.show()
