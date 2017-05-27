
import numpy as np
from sympy import Matrix, pprint
import matplotlib.pyplot as plt
import sys
data = np.zeros((20,2))
for i in range(20):
	data[i,0] = 2**(i+2)
	x = np.linspace(-1,1,2**(i+2))
	A = np.array([x**0,x**1,x**2,x**3]).T
	LP = np.array([x**0,x**1,(3./2.)*x**2-(1./2.),(5./2.)*x**3-(3./2.)*x]).T
	[q,r] = np.linalg.qr(A, mode = 'reduced')

	q = q/q[-1,:]
	residual = LP-q
	data[i,1] = np.max(residual)

plt.figure()
plt.plot(data)
plt.xlabel("$\Delta x$")
plt.ylabel("Max Error")
plt.savefig("test.png")
plt.show()
