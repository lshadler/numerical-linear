import numpy as np
import matplotlib.pyplot as plt
import sys
from sympy import pprint,Matrix

"""
mgs

Computes the modified Gram-Schmidt and furthermore
the Reduced QR Factorization of a given matrix

Args:
	matrix		An m x n matrix to be orthogonalized
Returns:
	Q			A unitary matrix 
	R			An upper triangular matrix
"""
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
		

"""
Main program

takes an argument of a stringsimilar to Matlab syntax

'1 0; 0 1'  <-->  1 0
				  0 1

Prints the QR factorization
"""

matrix = np.matrix(str(sys.argv[1:]))
print("---------- Original Matrix ----------")
pprint(Matrix(matrix))
q,r = mgs(matrix)
print("\n\n---------- Gram-Schmidt Result ----------\n\n")
print("---------- Q ----------")
pprint(Matrix(q))
print("\n\n---------- R ----------")
pprint(Matrix(r))


	
