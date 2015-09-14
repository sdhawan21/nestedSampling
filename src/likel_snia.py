"""
Using Evidence output from nested sampling, distinguish between different cosmological models by setting criteria for Bayes Factor values

Using nested (K. Barbary) the aim is to have a robust analyiss of alternative cosmological models that claim to be better or as good fits as LCDM.
"""

import numpy as np
import scipy
import nestle as nt
import matplotlib.pyplot as plt 
import sys

from pack import dist
fil=sys.argv[1]
c=2.9972e5
M=-19.09
sn=np.loadtxt(fil, usecols=(1, 2, 3), skiprows=1)


def lum_dist(z, om, ol, h0=70):
	"""
	expression for luminosity distance (currently not used)
	"""
	q0=(om/2)-ol
	try:
		a=z*(1-q0)/(np.sqrt(1+2*q0*z)+1+q0*z )
		dl=(1+a)*c*z/h0
		return dl
	except:
		return np.inf

def lnlikel(theta):
	"""
	Define chi_sq likelihood
	
	"""
	
	sn1=sn[2:3]
	z=sn1[:,0]; mag=sn1[:,1]; sig=sn1[:,2]
	
	#dl=lum_dist(z, theta[0], theta[1])
	
	#uses cosmocalc distance measurement
	#model=dist.mod(z)
	model= z**2
	return -0.5*np.sum((mag-model)**2/sig**2.)

def prior(u):
    """
    define prior on omega_lam and omega_m
    """
    return 2.0*u
    
    
res=nt.nest(lnlikel, prior, 2)
