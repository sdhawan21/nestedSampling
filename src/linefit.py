"""
Nested sampling for fitting a line observed data and obtaining the models

Shaw et al. 2007
David Mackay and John Skilling's awesome page

And Kyle barbary's kindness to put up Nestle as a public repo
"""

import nestle as nt

import numpy as np
import matplotlib.pyplot as plt
import triangle
import sys

from time import time
from pack.hpd import hpd

infile=np.loadtxt(sys.argv[1], usecols=(1, 2, 3, 6, 5), skiprows=1)

def model(theta, x):
    """
    define a simple linear model
    """
    m, c = theta
    return m*x + c

def lnlike(theta):
	"""
	chisq likelihood for model
	"""
	x=infile[:,-1] 

	y=infile[:,0]; yerr=infile[:,1]
	
	#model=theta[0]*x+theta[1]
	return -0.5*(np.sum((y-model(theta, x))**2/yerr**2))

#likelihood for quadratic fit (just to show that the non-linearity in the different regimes is not significant)	
def lnlike_poly(theta):

	x=infile[:,-1] 

	y=infile[:,0]; yerr=infile[:,1]
	
	model=theta[0]*x**2+theta[1]*x+theta[2]
	return -0.5*(np.sum((y-model)**2/yerr**2))

def prior(theta):
    return np.array([10.0, 20.0])*theta+np.array([-5.0, -10.0])

def prior1(x):
    """

	Uniform prior, this maps x=[0, 1) to m, c  in -5, 5
    """
    return 10.0*x-5.0
	
start=time()

#perform sampling
res = nt.nest(lnlike, prior, 2,  nobj=1000, maxiter=100000)

#perform sampling for different model
res_poly=nt.nest(lnlike_poly, prior1, 3,  nobj=1000, maxiter=100000)

end=time()

#prints output, can compare evidence

print "\nEvidence is:", res.logz
print "Best fit slope value:\t ",	np.median(res.samples[:,0]), np.median(res.samples[:,1])
print "\nEvidence for degree 2 polynomial is: \t", res_poly.logz

print "\nTime it took (in seconds) \t:", end-start

#use the fabulous triangle plot machinery to show the samples 
fig=triangle.corner(res.samples, weights=res.weights, labels=["m", "b"])
plt.show()

