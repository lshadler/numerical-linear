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
	Q			A matrix with unitary columns
	R			An upper triangular matrix
"""
def house(matrix):
	m,n = np.shape(matrix)
	# Instantiate matrices
	V = np.zeros((m,n))
	A = np.copy(matrix)
	R = np.zeros((n,n))


	# 	CREATE R MATRIX
	for k in range(n):
		x = np.copy(A[k:m,k:k+1])
		e1 = np.eye(m-k,1,0)
		updateVect = np.sign(x[0]) * np.linalg.norm(x) * e1
		v = x + updateVect
		v = v / (1.*np.linalg.norm(v))
		V[k:m,k:k+1]  += v
		A[ k:m, k:n ] -= 2*np.dot(v,np.dot(v.T,A[k:m,k:n]))
		#print("Iter {}:".format(k+1))
		#pprint(Matrix(updateMat))

	Q = formQ(V)
	return Q,A,V
	
	# FORM Q MATRIX
def formQ(V):

	m,n = np.shape(V)
	Q = np.eye(m,m)
	for k in range(n):
		i = n-k-1
		Vkk = V[i:m,i:i+1]
		Q[i:m,:] = Q[i:m,:] - 2*Vkk*np.dot(Vkk.T,Q[i:m,:])
	return Q
		

"""
Main program

takes an argument of a stringsimilar to Matlab syntax

'1 0; 0 1'  <-->  1 0
				  0 1

Prints the QR factorization
"""
def main():
	matrix = np.matrix(str(sys.argv[1:]))
	print("---------- Original Matrix ----------")
	pprint(Matrix(matrix))
	q,r = house(matrix)
	print("\n\n---------- Householder Result ----------\n\n")
	print("\n\n---------- Q ----------")
	pprint(Matrix(q))
	print("\n\n---------- R ----------")
	pprint(Matrix(r))
	print("\n\n---------- Reconstructed Matrix ----------")
	pprint(Matrix(np.dot(q,r)))


	
