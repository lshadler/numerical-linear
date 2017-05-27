import numpy as np
import scipy.linalg as spl
import matplotlib.pyplot as plt
"""
Observations of the 2-norm of positive-non-definite matrices
"""
plt.figure()
for i in range(10):
	A = np.random.randn(10,10) - 2*np.eye(10,10)
	ew,ev = np.linalg.eig(A)
	specA = np.real(np.max(ew))
	ts = np.linspace(1,20,10000)
	tss = []
	norms = []
	for t in ts:
		tss.append(t)
		norms.append( np.linalg.norm(spl.expm(t*A),ord=2) )
	alphas = np.exp(ts * specA)

	plt.semilogy(ts,alphas)
	plt.semilogy(tss,norms)
	plt.title('2-Norm')
	plt.ylabel('$||e^{tA}||_2$')
	plt.xlabel('t')
plt.show()

