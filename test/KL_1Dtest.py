# Test the 1D Karhunen-Loeve simulator: Generate Brownian motion
from GaussRF.GaussRF import GaussF_KL1D
import numpy as np
import matplotlib.pyplot as plt
# parameters of the BM
a = 0. # interval endpoints a and b
b = 1.
N = 99 # order of expansion
M = 100 # number of grid points
def Bm_cov(s, t): #Define Brownian motion
    return np.minimum(s, t)
# Generate Brownian motion KL object
X_sim = GaussF_KL1D( N = 99, M = 100, a = 0., b = 1.,
                     Cov = Bm_cov, method = "KL_EOLE", samples = 1)
#Check the eigenvalues and -functions
phi, L = X_sim.eigens()
#plot the eigenvlaues: pi/L = (k -0.5)**2 for BM
L_ex = [(k+0.5)**2 for k in range(10)]
L_app = 1./(L[:10]*np.pi**2)
plt.plot(L_ex, label = "exact eigenvalues")
plt.plot(L_app,'x', label = "numerical eigenvalues")
plt.legend()
plt.ylabel(r' $\frac{1}{\lambda_k\pi^2}$')
plt.title(' Eigenvalues')
plt.show()
#plot the eigenfunctions
tgrid = X_sim.grid()
for i in range(6):
    plt.subplot(2, 3, i + 1).set_title(r' $\phi_k$, k = {}'.format(i + 1))
    exact = np.sqrt(2)*np.sin((i + 0.5) * np.pi * tgrid) # exact eigenfunc
    apprx = np.abs(phi[:, i]) * np.sign(exact) # approximate eigen, same sign as exact
    plt.plot(tgrid, apprx, 'x', label = 'numerical')
    plt.plot(tgrid, exact, label = 'exact')
    plt.legend()
plt.show()
# Get the Brownian motion simulation as an array
X = X_sim.Gfield()
plt.plot(X)
plt.show()

