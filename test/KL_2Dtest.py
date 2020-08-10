# Test the 2D Karhunen-Loeve simulator: check exponential error
from GaussRF.GaussRF import GaussF_KL2D
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
#Parameters of the 2D exponential covariance
method = "KL_EOLE"
a, b, c, d = 0., 1., 0., 1.
M_vals = [11, 20, 50]
def K2D(x, y):
    return np.exp(-np.linalg.norm(x - y, 1, axis = 1))
#Functions for finding eigenvalues
def f(x):
    return (x**2 - 1)*np.tan(x) -2*x
def fprime(x):
    return -2 + 2*x*np.tan(x) +(x**2-1)/(np.cos(x)**2)

# get the eigenvalues of the standard exponential covariance
N= 10 # number of eigenvalues desired
L_ref = []
w_roots = [] 
N = 10 # number of eigenvalues desired
M_vals = [11,20,50] # differing discretization sizes
for k in range(N): # find the roots that provide the eigenvalues
    w0 = 1.2 + k*np.pi # initial guess for the Newton method
    w = optimize.newton(f,w0)
    w_roots.append(w)
L_ref = 2./( 1 + np.array(w_roots)**2) # 1D Evals
L_refx, L_refy = np.meshgrid(L_ref, L_ref) # prepare tensor of Evals
L_ref = (L_refx*L_refy).flatten() # tensor product eigenvalues
arg_sort = np.argsort(-L_ref)
L_ref = L_ref[arg_sort][:N] # eigenvalues in descending order.

#calculate relative error variance
err_var_ref = 1 - np.sum(L_ref) / ((b - a) * (d - c))
rel_err_var = np.zeros(len(M_vals))
counter = 0 # iterator for relative error index
for M in M_vals:
    phi_comp, L_comp = GaussF_KL2D(N, M, M, [a, b, c, d], K2D, method).eigens()
    err_var_comp = 1 - np.sum(L_comp) / ((b - a) * (d - c))
    rel_err_var[counter] = np.abs(err_var_comp - err_var_ref) / err_var_ref
    counter += 1
plt.loglog(M_vals, rel_err_var, 'o--', label ='relative error variance')
plt.plot([10,10*np.sqrt(10)],[0.12,0.012], 'x--',color= 'r', label = 'reference slope m=-2')
plt.title(r'2D exponential covariance $\varepsilon_{rel}$, ' + method + ' method')
plt.legend()
plt.show()
