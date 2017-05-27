import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix, pprint
"""
Successive rank approximations of "Hello"
"""
def makeHelloMatrix():
	mat = np.zeros((15,40))
	# H
	mat[2:10,2:4]   = 1
	mat[5:7,4:6]    = 1
	mat[2:10,6:8]   = 1

	# E
	mat[3:11,10:12]  = 1
	mat[3:5,11:16]   = 1
	mat[6:8,11:16]   = 1
	mat[9:11,11:16]  = 1

	# L
	mat[4:12,18:20]  = 1
	mat[10:12,20:24] = 1
	# L
	mat[5:13,26:28]  = 1
	mat[11:13,28:32] = 1

	# O
	mat[6:14,34:36]  = 1
	mat[6:14,38:40]  = 1
	mat[6:8,36:38]   = 1
	mat[12:14,36:38] = 1
	return mat

def makeAssMatrix():
	mat = np.zeros((15,40))

	# A
	mat[2:12,2:4] 		= 1
	mat[2:12,6:8]		= 1
	mat[2:4,4:6]		= 1
	mat[6:8,4:6]		= 1

	# S
	mat[3:5,10:16]		= 1
	mat[3:9,10:12]		= 1
	mat[7:9,10:16] 		= 1
	mat[7:13,14:16] 	= 1
	mat[11:13,10:16]	= 1
	
	# S
	mat[4:6,18:24]		= 1
	mat[4:10,18:20]		= 1
	mat[8:10,18:24] 	= 1
	mat[8:14,22:24] 	= 1
	mat[12:14,18:24]	= 1


	# Smile

	mat[4:7,28:30]		= 1
	mat[4:7,34:36]		= 1
	mat[11:13,29:35]	= 1
	mat[9:11,27:28]		= 1
	mat[10:12,28:29]	= 1
	mat[10:12,35:36] 	= 1
	mat[9:11,36:37]		= 1
	return mat



mat = makeAssMatrix()

u,sp,v = np.linalg.svd(mat)

plt.figure()
plt.plot(sp)
plt.title("Singular Values")
plt.savefig("singVals.png")
s = np.zeros((15,40))
for i in range(len(sp)):
	s[i,i] = sp[i]

print("Singular Values")
pprint(Matrix(s))
print("U")
pprint(Matrix(u))
print("V")
pprint(Matrix(v))

fig =plt.figure()
for i in range(len(sp)):
	u_t = u[:,:(i+1)]
	s_t = s[:(i+1),:(i+1)]
	v_t = v[:(i+1),:]
	
	mat_t = np.dot(u_t,s_t)
	mat_t = np.dot(mat_t,v_t)
	ax = fig.add_subplot(8,2,i+1)
	ax.set_aspect('equal')
	ax.set_xticks([])
	ax.set_yticks([])
	plt.ylabel("$r = {}$".format(i+1))
	plt.imshow(mat_t, interpolation='nearest', cmap='jet')
	filename = "rankApprox.png".format(i+1)
plt.savefig(filename)
plt.show()
