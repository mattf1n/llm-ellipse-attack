import time
import numpy as np
import matplotlib.pyplot as plt

solve_times = []
# lstsq_times = []
for exponent in range(14):
    size = pow(2, exponent)
    A = np.random.random((size, size))
    solve_start = time.time()
    np.linalg.solve(A, np.ones(size))
    solve_times.append(time.time() - solve_start)
    # lstsq_start = time.time()
    # np.linalg.lstsq(A, np.ones(size), rcond=None)
    # lstsq_times.append(time.time() - lstsq_start)

plt.plot(solve_times, label="Solve")
# plt.plot(lstsq_times, label="Lstsq")
plt.legend()
plt.show()
