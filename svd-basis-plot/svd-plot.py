
"""
@author Lucas Shadler

Simple function to plot the dimensions 
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

def main():


	# Take command line arguments
	if len(sys.argv) == 5:
		p   = [float(i) for i in sys.argv[1:]]
		try:
			mat = np.array([[p[0],p[1]],[p[2],p[3]]])
		except Exception as e:
			print("Please define a 2x2 matrix...")
			sys.exit()
	else:
		print("Usage: svd-basis-plot.py [a11 a12 a21 a22]") 
		print("Demonstrating on identity matrix")
		mat   = np.array([[1,0],[0,1]])

	# Perform Singular Value Decomposition
	u,s,v = np.linalg.svd(mat)

	# Plot the two ellipse with vectors
	plotEllipse(u,s,title= "U-Space")
	print("\n\nU matrix:")
	print(u)
	plotEllipse(v,  title= "V-Space")
	print("\n\n V matrix:")
	print(v)
	plt.show()

def plotEllipse(mat,s = np.array([1,1]),title= "Matrix plot"):
	
	# Extract individual vector coordinates
	x1,y1 = float(s[0]*mat[0,0]),float(s[0]*mat[1,0])
	x2,y2 = float(s[1]*mat[0,1]),float(s[1]*mat[1,1])

	# Find magnitude of major and minor axes
	a = np.sqrt(x1**2+y1**2)
	b = np.sqrt(x2**2+y2**2)

	# Calculate the rotation angle
	if x1 == 0:
		theta = np.pi/2.0
	else:
		theta = np.arctan(y1/x1)

	# Populate the ellipse parametric points
	t = np.linspace(0,2*np.pi,10000)
	x = a*np.cos(t)*np.cos(theta) - b*np.sin(t)*np.sin(theta)
	y = a*np.cos(t)*np.sin(theta) + b*np.sin(t)*np.cos(theta)

	# Plot the figure
	plt.figure()
	ax = plt.gca()
	ax.quiver([0 ,0 ],[0 ,0 ]     ,
			  [x1,x2],[y1,y2]     ,
			  angles = 'xy'       ,
			  scale_units = 'xy'  , 
			  scale=1             )
	plt.plot(x,y)
	plt.title(title)


if __name__ == "__main__":
	main()

