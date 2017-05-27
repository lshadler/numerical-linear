import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix, pprint
import sys
from PIL import Image,ImageOps




def getSingleColorMat(u,s,v,i):
	u_t = u[:,:(i+1)]
	s_t = s[:(i+1),:(i+1)]
	v_t = v[:(i+1),:]
	
	mat_t = np.dot(u_t,s_t)
	mat_t = np.dot(mat_t,v_t)
	return mat_t


print(sys.argv[1])
im = Image.open(sys.argv[1])
m,n = im.size
#im.show()
mat = np.array(ImageOps.invert(im))
r,g,b = mat[:,:,0],mat[:,:,1],mat[:,:,2]

fig =plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(mat, interpolation='nearest')
"""
fig =plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(r, interpolation='nearest')

fig =plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(g, interpolation='nearest')

fig =plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_aspect('equal')
plt.imshow(b, interpolation='nearest')
"""

ur,spr,vr = np.linalg.svd(r)
ug,spg,vg = np.linalg.svd(g)
ub,spb,vb = np.linalg.svd(b)

sr = np.zeros((m,n))
sg = np.zeros((m,n))
sb = np.zeros((m,n))


for i in range(n):
	sr[i,i] = spr[i]
	sg[i,i] = spg[i]
	sb[i,i] = spb[i]

for i in np.arange(0,n,50):
    r_t = getSingleColorMat(ur,sr,vr,i)
    g_t = getSingleColorMat(ug,sg,vg,i)
    b_t = getSingleColorMat(ub,sb,vb,i)
    mat_t = np.zeros((n,m,3))
    #mat_t[:,:,0] = r_t
    #mat_t[:,:,1] = g_t
    #mat_t[:,:,2] = b_t
    for i in range(n):
    	for j in range(m):
    		mat_t[i,j] = (r_t[i,j],g_t[i,j],b_t[i,j])
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.ylabel("$r = {}$".format(i+1))
    plt.imshow(mat_t,interpolation='nearest')
plt.show()


