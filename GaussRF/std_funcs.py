"""Collection of standard covariance functions"""
import warnings
import numpy as np
from scipy.special import kv, gamma 

def exp_cov1D(l = 1., p = 1, sigma = 1.):
    '''
    Return a 1-D exponential covariance function of
    length scale l, exponential 1 and power p. Variance sigma^2
    '''
    def f(s, t):
        return sigma**2*np.exp(-np.abs(s - t)**p / l)
    return f
def RQ_cov1D(alpha = 1., l = 1., sigma = 1.):
    '''
    Return a 1-D rational quadratic (RQ) covariance function
    of shape parameter alpha and length scale l. Variance sigma^2
    '''
    def f(s, t):
        return sigma**2 * ( 1 + np.abs(s - t)**2 / (2 * alpha * l**2) )**(-alpha)

def Mat_cov1D(nu = 1. , sigma = 1., l = 2.):
    '''
    Return a 1-D Matern (Mat) covariance function
    of parameters l and nu. Variance sigma^2
    '''
    def f(s, t):
        d = np.abs(s - t)
        Rho = sigma**2 * np.ones(d.shape) # implicitly deals with origin special case
        Rho[ d > 1e-10] *= 2**(1 - nu) * (np.sqrt(2 * nu) * d[ d > 1e-10] / l) * kv(np.sqrt(2 * nu) * d[ d > 1e-10] / l)
        return Rho
def exp_cov2D(l1 = 1., l2 = 1., p = 1, sigma = 1.):
    '''
    Return a 2-D exponential covariance function of length scale l1 in first dimension,
    length scale l2 in the second dimesion and power p. Variance sigma^2
    '''
    def f(s, t):
        s[:,0], t[:,0] = s[:, 0] / l1, t[:, 0] / l1 # x-length scale
        s[:,1], t[:,1] = s[:, 1] / l2, t[:, 1] / l2 # y-length scale
        r = np.sqrt( np.linalg.norm( s - t, p, axis = 1) ) # argument of exp
        return sigma**2*np.exp(-r)

def RQ_cov2D(aplha = 1., l1 = 1., l2 = 1., sigma = 1.):
    '''
    Return a 2-D rational quadratic (RQ) covariance function
    of shape parameter alpha, length scale l1 in the first dim
    and length scale l2 in the second dim. Variance sigma^2
    '''
    def f(s, t):
        s[:, 0], t[:, 0] = s[:, 0] / l1, t[:, 0] / l1 #x-length scale
        s[:, 1], t[:,1] = s[:, 1] / l2, t[:, 1] / l2 #y-length scale
        return sigma**2 * ( 1 + np.abs( s - t)**2 / (2 * alpha * l**2) )**(-alpha)

def Mat_cov2D(nu = 1., l1 = 1., l2 = 1., p = 1, sigma = 1.):
    '''
    Return a 2-D Matern (Mat) covariance function of parameter
    nu, lengthscales l1 and l2, and power p. Variance sigma^2
    '''
    def f(s, t):
        s[:, 0], t[:, 0] = s[:, 0] / l1, t[:, 0] / l1 # x-length scale
        s[:, 1], t[:, 1] = s[:, 1] / l2, t[:, 1] / l2 # y- length scale
        r = np.sqrt( np.linalg.norm( s - t, p, axis = 1) ) # distance function
        Rho = sigma**2 * np.ones(r.shape) # implicitly deal with origin special case
        Rho[ d > 1e-10] *= 2**(1 - nu) * (np.sqrt(2 * nu) * d[ d > 1e-10]) * kv(np.sqrt(2 * nu) * d[ d > 1e-10])
        return Rho
    
    
