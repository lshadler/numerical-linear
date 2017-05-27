import numpy as np
import matplotlib.pyplot as plt
import sys
from sympy import pprint,Matrix
from house import *
from mgs import *

M 	 		 = 50
N 			 = 12
t 			 = np.linspace(0,1,50)
vanderMatrix = np.matrix(np.fliplr(np.vander(t,N)))


def backSub(r,b):
	m,n = np.shape(r)
	x   = np.zeros((n,1))
	for i in range(n):
		k = m-i-1
		x[k] = b[k]
		for j in range(k,n):
			x[k] -= x[k]*r[k,j]
		x[k] = x[k] / r[k,k]
	return x
	
def forwardSub(r,b):
	m,n = np.shape(r)
	x   = np.zeros((n,1))
	for k in range(n):
		x[k] = b[k]
		for j in range(1,k):
			x[k] -= x[k]*r[k,j]
		x[k] = x[k] / r[k,k]
	return x

def choleskyFact(A):
	m,n = np.shape(A)
	R = np.copy(A)
	for k in range(n):
		for j in range(k+1,n):
			R[j,j:m] -= R[k,j:m]*R[k,j]/R[k,k]
		R[k,k:m] = R[k,k:m]/np.sqrt(R[k,k])
	return R
	
cosineOnT    = np.cos(4*t)

# NORMAL EQUATIONS

a2 = np.dot(vanderMatrix.conj().T,vanderMatrix)
ab = np.dot(vanderMatrix.conj().T,cosineOnT)
xnorm = np.linalg.solve(a2,ab.T)

# MODIFIED GRAM SCHMIDT
qm,rm		= mgs(vanderMatrix)
iV = np.dot(qm.conj().T,cosineOnT)
xm = backSub(rm,iV)

# HOUSEHOLDERS QR
qh,rh,wh	= house(vanderMatrix)
rh = rh[0:N,0:N]
iV = np.dot(qh.conj().T,cosineOnT)
xh = backSub(rh,iV)

# PYTHONS QR FACTORIZATION

qp,rp	= np.linalg.qr(vanderMatrix)
rp      = rp[0:N,0:N]
iV 		= np.dot(qp.conj().T,cosineOnT)
xp 		= backSub(rp,iV.T)


# SOLVING ALGORITHM

xlq,res,rank,s = np.linalg.lstsq(vanderMatrix,cosineOnT)
#print(xlq)

# SVD
u,s,v = np.linalg.svd(vanderMatrix)
ess = np.zeros((12,50))
for i in range(len(s)):
	ess[i,i] = 1/s[i]
w = np.dot(ess,np.dot(u.conj().T,cosineOnT).T)
xs = np.dot(v.conj().T,w)

fits = np.zeros((12,6))
fits[:,0:1] = xnorm
fits[:,1:2] = xm
fits[:,2:3] = xh
fits[:,3:4] = xp
fits[:,4] = xlq
fits[:,5:6] = xs
pprint(Matrix(fits))
