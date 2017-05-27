import numpy as np
from sympy import Matrix, pprint
import matplotlib.pyplot as plt
import sys
if(len(sys.argv) > 1):
	x = np.linspace(-1,1,2**float(sys.argv[1]))
else:	
	x = np.linspace(-1,1,256)
A = np.array([x**0,x**1,x**2,x**3]).T
LP = np.array([x**0,x**1,(3./2.)*x**2-(1./2.),(5./2.)*x**3-(3./2.)*x]).T
[q,r] = np.linalg.qr(A, mode = 'reduced')
q = q/q[-1,:]
if(sys.argv == 1):
	pprint(Matrix(q))
	pprint(Matrix(r))

plt.figure()
plt.plot(x,q)
plt.title("Approximate $P_k$")
plt.savefig("ApproxLeg.png")
plt.figure()
plt.plot(x,(q-LP))
plt.title("Error")
plt.savefig("error.png")
plt.show()
