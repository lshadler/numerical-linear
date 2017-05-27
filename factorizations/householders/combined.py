import numpy as np
import matplotlib.pyplot as plt
import sys
from sympy import pprint,Matrix

def house(matrix):
	m, n = np.shape(matrix)
	# Instantiate matrices
	V = np.zeros((m,n))
	A = np.copy(matrix)
	R = np.zeros((n,n))

	if m < n:
		print("IndexError: m < n")
	# 	CREATE R MATRIX
	for k in range(n):

		x = np.copy(A[k:m,k:k+1])
		e1 = np.eye(m-k,1,0)
		updateVect = (np.sign(x[0]) * np.linalg.norm(x) * e1)
		v = x + updateVect
		v = v / (1.*np.linalg.norm(v))
		V[k:m,k:k+1]  += v
		A[ k:m, k:n ] -= 2.0*np.dot(v,np.dot(v.T,A[k:m,k:n]))

		#print("Iter {}:".format(k+1))
		#pprint(Matrix(updateMat))
	
	Q = np.eye(m,m)

	# FORM Q MATRIX
	for k in range(n):
		i = n-k-1
		Vkk = V[i:m,i:i+1]
		Q[i:m,:] = Q[i:m,:] - 2*Vkk*np.dot(Vkk.T,Q[i:m,:])
	return Q,A



def mgs(matrix):
	m,n = np.shape(matrix)
	# Instantiate matrices
	V = np.zeros((m,n))
	Q = np.zeros((m,n))
	R = np.zeros((n,n))

	# Hard copy of original matrix into V
	for i in range(n):
		V[:,i] = matrix.A[:,i]

		
	for i in range(n):
		R[i,i] = np.linalg.norm(V[:,i])			# Compute diagonal element of R
		Q[:,i] = V[:,i]/R[i,i]					# Compute Q_i vector
		for j in range(i+1,n):					# Update subsequent vectors	
			R[i,j] = np.dot(Q[:,i],V[:,j])		# Copute non-diagonal elements
			sub = R[i,j]*Q[:,i]					# Component to remove from V_j
			V[:,j] = V[:,j] - (R[i,j]*Q[:,i])	# Subtract from V_j
	return Q,R

if len(sys.argv) < 2:
	matrix = np.matrix(np.random.rand(3,3))
else:
	matrix = np.matrix(str(sys.argv[1:]))
print("---------- Original Matrix ----------")
pprint(Matrix(matrix))
q1,r1 = house(matrix)
q2,r2 = mgs(matrix)
q3,r3 = np.linalg.qr(matrix)
print("\n\n---------- Q ----------")
pprint(Matrix(q1))
pprint(Matrix(q2))
pprint(Matrix(q3))
print("\n\n---------- R ----------")
pprint(Matrix(r1))
pprint(Matrix(r2))
pprint(Matrix(r3))
"""
print("\n\n\n\n\n\n\n---------- Residual ----------")
print("\n\n---------- Q ----------")
pprint(Matrix(np.abs(q1)-np.abs(q3)))
pprint(Matrix(np.abs(q2)-np.abs(q3)))
print("\n\n---------- R ----------")
pprint(Matrix(np.abs(r1)-np.abs(r3)))
pprint(Matrix(np.abs(r2)-np.abs(r3)))
"""
