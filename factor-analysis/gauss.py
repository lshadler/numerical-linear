import numpy as np
import matplotlib.pyplot as plt
import sys
from sympy import pprint,Matrix

"""
"""
def gauss_pp(A,b):
	n =  len(A)
	if b.size != n:
		raise ValueError("Invalid argument: incompatible sizes between A & b.", b.size, n)
	# k represents the current pivot row. Since GE traverses the matrix in the upper 
	# right triangle, we also use k for indicating the k-th diagonal column index.
	for k in range(n-1):
		#Choose largest pivot element below (and including) k
		maxindex = abs(A[k:,k]).argmax() + k
		if A[maxindex, k] == 0:
			raise ValueError("Matrix is singular.")
		#Swap rows
		if maxindex != k:
			A[[k,maxindex]] = A[[maxindex, k]]
			b[[k,maxindex]] = b[[maxindex, k]]
		for row in range(k+1, n):
			multiplier = A[row][k]/A[k][k]
			#the only one in this column since the rest are zero
			A[row][k] = multiplier
			for col in range(k + 1, n):
				A[row][col] = A[row][col] - multiplier*A[k][col]
			#Equation solution column
			b[row] = b[row] - multiplier*b[k]
	#pprint(Matrix(A))
	#pprint(Matrix(b))
	x = np.zeros(n)
	k = n-1
	x[k] = b[k]/A[k,k]
	while k >= 0:
		x[k] = (b[k] - np.dot(A[k,k+1:],x[k+1:]))/A[k,k]
		k = k-1
	return x
"""
Tests the effective error of Gaussian Elimination iterative method
	on a special matrix
"""
N = 1000
x_error = []
for i in range(N):
	A = np.eye(60,60)
	for i in range(60):
		for j in range(60):
			if i>j:
				A[i,j] = -1 
			if(j == 59):
				A[i,j] = 1
	noise  = np.random.normal(0.0,0.0001,np.shape(A))
	A = A+noise
	x_rand = np.random.rand(60,1)
	b_calc = np.dot(A,x_rand)
	x_sol  = gauss_pp(A,b_calc)
	x_error.append(np.mean(np.abs(x_sol-x_rand)))
print(x_error)
fig = plt.figure()
plt.plot(x_error)
plt.xlabel("Trial Number")
plt.ylabel("Average Absolute Error")
plt.title("Mean: {} Std: {}".format(np.mean(x_error),np.std(x_error)))
plt.show()


	
