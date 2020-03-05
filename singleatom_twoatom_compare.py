import matplotlib.pyplot as plt
import numpy as np

from singleatom_numerical import solve_with as solve_singleatom
from twoatom_nointeraction_numerical import solve_with as solve_twoatom

sol1 = solve_singleatom()
sol2 = solve_twoatom()

plt.plot(sol1.t, np.abs(sol1.y[0])**2, label='$|c_1|^2$')
plt.plot(sol1.t, np.abs(sol1.y[1])**2, label='$|c_r|^2$')


# Combined Prob. Amps
plt.plot(sol2.t, np.abs(sol2.y[0])**2 + np.abs(sol2.y[2]) ** 2,
         label='$|c_{11}|^2 + |c_{r1}|^2$', ls='--')
plt.plot(sol2.t, np.abs(sol2.y[1])**2 + np.abs(sol2.y[3]) ** 2,
         label='$|c_{1r}|^2 + |c_{rr}|^2$', ls='--')

plt.xlabel('Time, $t$')
plt.ylabel('Probability Amplitude')
plt.legend()
plt.show()