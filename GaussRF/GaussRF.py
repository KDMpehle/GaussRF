"""Generate random fields in one and two rectangular dimensions."""
import warnings, inspect
import numpy as np
import scipy.sparse.linalg as spla


class GaussF_KL1D(object):
    """ The class for a one-dimensional gaussian field
    i.e a stochastic process.

    The process is instantiated with KL order N, M increments,
    covariance Cov, a method and sample size samples.
    """
    def __init__(self, N, M, a, b, Cov, method = "KL_EOLE", samples = 1):
        # instantiate the 1D random field/Gaussian process
        self.M = M
        self.N = N
        self.a = a
        self.b = b
        self.Cov = Cov
        self.samples = samples
        self._eigens = self.eigens# method to get the eigenvalues and vectors
        self._grids = {"KL_EOLE": self._eolegrid, "KL_gaussleg": self._gaussleggrid}
        self._weights = {"KL_EOLE": self._eoleweights, "KL_gaussleg": ''} 
        self.method = method
        self._grid = self._grids[self.method]
        self._weight = self._weights[self.method] # function to get the weights matrix for a method

        self._changed = False # a flag for whether variables have been changed
    def __str__(self):
        # Str method.
        return (
            "1D KL expansion ("
            + self.method
            + ") and covariance "
            + self.Cov.__name__
            +"on ["
            + str(self.a) 
            + ", "
            + str(self.b) 
            + "] with KL order "
            + str(self.N) 
            + " and "
            + str(M)
            + " grid points and "
            + str(self.samples)
            + " samples."
            )
    
    def __repr__(self):
        # Repr method.
        return (
            "GaussF_KL1D(N = "
            + str(self.N)
            + ", M = "
            + str(self.M)
            + ", a = "
            + str(self.a)
            + ", b = "
            + str(self.b)
            + ", Cov = "
            + self.Cov.__name__
            + ', method = "'
            + self.method
            + '", samples = "'
            + str(self.samples)
            + ")"
            )
    # variable setters

    @property
    def M(self):
        """ Get the number of points in the domain."""
        return self._M
    
    @M.setter
    def M(self, value):
        if not isinstance(value, int) or value <= 0:
            raise TypeError("Number of points must be a positive integer.")
        self._M = value
        self._changed = True

    @property
    def N(self):
        """ Get the order of the KL expansion."""
        return self._N

    @N.setter
    def N(self, value):
        if not isinstance(value, int) or value <= 0 :
            raise TypeError("Order of the KL expansion must be a positive integer.")
        elif value >= self.M:
            raise ValueError("N must be less than M. Lanczos algorithm of eigsh can't get all eigenvectors")
        self._N = value
        self._changed = True

    @property
    def a(self):
        """ left end point."""
        return self._a

    @a.setter
    def a(self, value):
        if not ( isinstance(value, (int, float)) and not isinstance(value, bool) ):
            raise TypeError("a needs to be real")
        self._a = value
        self._changed = True
        
    @property
    def b(self):
        """ right end point."""
        return self._b

    @b.setter
    def b(self, value):
        if not ( isinstance(value, (int, float)) and not isinstance(value, bool)):
            raise TypeError("b needs to be real")
        elif value <= self.a:
            raise ValueError(" b needs to be larger than a, [a, b]")
        self._b = value
        self._changed = True
        
    @property
    def Cov(self):
        """Covariance function."""
        return self._Cov
    
    @Cov.setter
    def Cov(self, value):
        try:
            num_args = len(inspect.getfullargspec(value)[0])
        except Exception:
            raise ValueError(" Covariance is a function")
        if not callable(value) or num_args != 2:
            raise ValueError(" Covariance must be a function of two arguments")
        self._Cov = value
        self._changed = True
        
    @property
    def method(self):
        """ Get the discretisation method chosen."""
        return self._method

    @method.setter
    def method(self, value):
        if value not in self._grids:
            raise ValueError(' Supported discretisations are "KL_EOLE" and "KL_gaussleg" ')
        self._method = value
        self._grid = self._grids[self.method]
        self._changed = True

    @property
    def samples(self):
        " Get the sample size."
        return self._samples

    @samples.setter
    def samples(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(" number of samples must be a positive integer")
        self._samples = value
        self._changed = True
    
    # methods of the class
    def Gfield(self):
        if self.method == "circembed":
            pass # Need to write this part.
        else:
            Z = np.random.randn(self.N, self.samples) # simulate standard normal
            X = np.zeros((self.M, self.samples)) # preallocate the sample array
            phi, L = self._eigens()
            for k in range(self.samples):
                for i in range(self.N):
                    X[:, k] += Z[i, k] * np.sqrt(L[i]) * phi[:, i] # Karhunen-Loeve expansion
            return X
        
    def eigens(self):
        if self.method == "KL_gaussleg":
            x, w = self._grid() # weights and grid for gauss legendre method
            W = 0.5 * (self.b - self.a) * np.diag(w) # get the weights matrix quickly here
            W_inv = np.sqrt(2. / (self.b - self.a)) * np.diag(1. / np.sqrt(w))
        else:
            x = self._grid() # get the grid points on which to evaluate the covariance
            W, W_inv = self._weight() # weights matrix and its sqrt inverse
            
        # Construct covariance matrix
        x1, x2 = np.meshgrid(x[: self.M], x[: self.M])
        C = self.Cov(x1, x2) # covariance matrix.
        B = np.dot(np.dot(np.sqrt(W), C), np.sqrt(W)) # symmetric B matrix
        
        # Get the eigenvalues and eigenvectors
        L, y = spla.eigsh(B, k = self.N)
        L, y = L[np.argsort(-L)].real, y[:, np.argsort(-L)].real # re-order the eigens
        phi = np.dot(W_inv, y) # original eigenvector problem.
        return phi, L # return the eigenvectors and values    
        
    def _eoleweights(self):
        # get weights associated with an EOLE simulation
        return (1. / self.M) * (self.b - self.a) * np.eye(self.M), np.sqrt(float(self.M) / (self.b - self.a)) * np.eye(self.M)

    def _eolegrid(self):
        # get the grid associated with an EOLE simulation
        return np.linspace(self.a, self.b, self.M + 1)
    
    def _gaussleggrid(self):
        # get the grid associated with the Gauss-legendre simulation
        xi, w = np.polynomial.legendre.leggauss(self.M)
        return ( (self.b - self.a) * xi / 2 + (self.a + self.b) / 2, w)# translate the GL points [a, b]
    
    def grid(self):
        # get the grid of the random field/ stochastic process
        if method == "KL_gaussleg":
            return self.grid()[0] # return just the grid of the gauss-Legendre points.
        return self._grid()

def GRfield_1D(N, M, a, b, Cov, method = "KL_EOLE", samples = 1):
    """ A simulation of a 1D Gauss random field array """
    RF = GaussF_KL1D(N, M, a, b, Cov, method, samples)
    return RF.Gfield()

def grid_1D(M, a, b, method):
    """ generate the times associated with the Random field. """
    if method == "KL_gaussleg":
        xi, w = np.polynomial.legendre.leggauss(M)
        return (b - a) * xi / 2 + (a + b) / 2
    else: # for now, the other grids are uniform grids
        return np.linspace(a, b, M + 1)

class GaussF_KL2D(object):
    """ The class for a two-dimensional field

    The field is instantiated with order N, number of
    x points n, number of y points m, endpoints, Sample siz
    [a, b, c, d] = [a, b] x [c, d], covariance Cov, a method
    and sample size samples.

    """

    def __init__(self, N, n, m, lims, Cov, method = "KL_EOLE", samples = 1):
        # instantiate the 2D Gaussian field
        self.n = n
        self.m = m
        self.N = N
        self.lims = lims
        self.a, self.b = lims[0:2]
        self.c, self.d = lims[2:4]
        self.A = (self.b - self.a) * (self.d - self.c)
        self.Cov = Cov
        self.samples = samples
        self._eigens = self.eigens
        self._grids = {"KL_EOLE": self._eolegrid, "KL_gaussleg": self._gaussleggrid}
        self._weights = {"KL_EOLE": self._eoleweights, "KL_gaussleg": ''} # need to pass the Gauss-legendre weights somehow. Not elegant 
        self.method = method
        self._grid = self._grids[self.method]
        self._weight = self._weights[self.method] # function to get the weights matrix for a method

    #do str method
    def __str__(self):
        # The str method
        return (
            "2D KL random field ("
            + self.method
            + "), covariance "
            + self.Cov.__name__
            + " on ["
            + str(self.a)
            + ", "
            + str(self.b)
            + "] x ["
            + str(self.c)
            + ", "
            + str(self.d)
            + "], with KL order "
            + str(self.N)
            + ", "
            + str(self.n)
            + " x-points and "
            + str(self.m)
            + " y-points, and sample size "
            + str(self.samples)
            )
    # do repr method
    def __repr__(self):
        # The Repr method.
        return (
            "GaussF_KL2D(N = "
            + str(self.N)
            + ", n = "
            + str(self.n)
            + ", m = "
            + str(self.m)
            + ", lims = "
            + str(self.lims)
            + ", Cov = "
            + self.Cov.__name__
            + ', method = "'
            + self.method
            + '", samples = '
            + str(self.samples)
            + ")" )
    # Variable setters and getters.
    @property
    def n(self):
        # Get number of x-points.
        return self._n

    @n.setter
    def n(self, value):
        if not isinstance(value, int) or value <= 0:
            raise TypeError(" Number of x-points must be a positive integer")
        self._n = value

    @property
    def m(self):
        # Get number of y-points.
        return self._m

    @m.setter
    def m(self, value):
        if not isinstance(value, int) or value <= 0:
            raise TypeError(" Number of y-points must be a positive integer")
        self._m = value

    @property
    def N(self):
        # Get the order of the KL expansion.
        return self._N

    @N.setter
    def N(self, value):
        if not isinstance(value, int) or value <= 0:
            raise TypeError(" Order of the KL expansion must be a positive integer.")
        elif value >= self.n * self.m:
            raise ValueError(" N must be less than n*m. Lanczos algortihm can't get all eigens")
        self._N = value

    @property
    def lims(self):
        # get the domain limits
        return self._lims

    @lims.setter
    def lims(self, value):
        if not isinstance(value, list):
            raise TypeError(" Limits must be provided as a list")
        elif len(value) != 4:
            raise TypeError(" Must provide 4 numbers for the limits")
        elif (value[1] <= value[0]) or (value[3] <= value[2]):
            raise ValueError(" Error in limits, a < b and c < d for rectangle [a, b] x [c, d]")
        self._lims = value
        self.a, self.b = self.lims[0:2]
        self.c, self.d = self.lims[2:4]
        self.A = (self.b - self.a) * (self.d - self.c)
        
    @property
    def Cov(self):
        # Get the covariance
        return self._Cov

    @Cov.setter
    def Cov(self, value):
        try:
            num_args = len(inspect.getfullargspec(value)[0])
        except Exception:
            raise ValueError("Covariance must be a python function")
        if not callable(value) or num_args != 2:
            raise ValueError("Covariance must be a function of two arguments")
        self._Cov = value

    @property
    def method(self):
        # Get the discretisation method chosen.
        return self._method

    @method.setter
    def method(self, value):
        if value not in self._grids:
            raise ValueError(' Supported discretisations are "KL_EOLE" and "KL_gaussleg"')
        self._method = value
        self._grid = self._grids[self.method]

    @property
    def samples(self):
        " Get the sample size"
        return self._samples

    @samples.setter
    def samples(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError(" Number of samples must be a positive integer")
        self._samples = value
        
    #methods of the class
    def Gfield(self):
        if self.method == "circembed":
            pass # write this section
        else:
            Z = np.random.randn(self.N, self.samples)
            X = np.zeros((self.n, self.m, self.samples)) # allocate array to hold sample
            phi, L = self._eigens()
            for k in range(self.samples):
                for i in range(self.N):
                    X[:,:,k] += np.sqrt(L[i]) * Z[i,k] * phi[:,i].reshape(self.n, self.m)
            return X
        
    def eigens(self):
        if self.method == "KL_gaussleg":
            x, y, w1, w2 = self._grid()
            W = (self.A / 4) * np.kron(np.diag(w1), np.diag(w2))
            W_inv = np.sqrt(4. / self.A) * np.sqrt(np.kron(np.diag(1. / w1), np.diag(1. / w2)))
            xx = np.hstack([np.repeat(x, self.m).reshape( self.n * self.m, 1),
                            np.tile(y, self.n).reshape(self.n * self.m, 1)])
        else:
            x, y = self._grid()
            W, W_inv = self._eoleweights()
            xx = np.hstack([np.repeat(x[: self.n], self.m).reshape( self.n * self.m, 1),
                            np.tile(y[: self.m], self.n).reshape(self.n * self.m, 1)])            
        # Construct covariance matrix 

        xxx = np.hstack([np.repeat(xx, self.n * self.m, axis = 0),
                         np.tile(xx, [self.n * self.m, 1])])
        C = self.Cov(xxx[:, 0:2], xxx[:, 2:]).reshape(self.n * self.m, self.n * self.m) # construct covariance matrix
        B = np.dot(np.dot(np.sqrt(W), C), np.sqrt(W)) # symmetric B matrix

        # Get spectral quantities
        L, y = spla.eigsh(B, k = self.N) #eigenvalues and vectors B.
        arg_sort = np.argsort(-L) # re-order eigens in descending order
        L, y = L[arg_sort].real, y[:, arg_sort].real
        phi = np.dot(W_inv, y) # get the original eigenvectors of C
        return phi, L
    
    def _eoleweights(self):
        # get the EOLE weight matrix and square-root inverse.
        return ( (self.A / (self.n * self.m)) * np.eye(self.n * self.m),
                 np.sqrt(self.n * self.m/self.A) * np.eye(self.n * self.m) )

    def _eolegrid(self):
        x1 = np.linspace(self.a, self.b , self.n + 1)
        x2 = np.linspace(self.c, self.d, self.m + 1)
        return (x1, x2)

    def _gaussleggrid(self):
        xi, w1 = np.polynomial.legendre.leggauss(self.n) # x points and weights
        zeta, w2 = np.polynomial.legendre.leggauss(self.m) # y points and weights
        x1 = 0.5 * (self.b - self.a) * xi + 0.5 * (self.a + self.b) # translate x-grid points
        x2 = 0.5 * (self.d - self.c) * zeta + 0.5 * (self.c + self.d) # translate y-coordinate points
        return (x1, x2, w1, w2)

def GRfield_2D(N, n, m, lims, Cov, method = "KL_EOLE", samples = 1):
    """ A simulation of a Gauss random field array """
    RF = GaussF_KL2D(N, n, m, lims, Cov, method, samples)
    return RF.Gfield()

def grid_2D(n, m, lims, method):
    # Get the coordinates
    a, b, c, d = lims
    if method == "KL_gaussleg":
        # Get the Gauss-Legendre points
        xi, w  = np.polynomial.legendre.leggauss(n)
        zeta, w = np.polynomial.legendre.leggauss(m)
        x = (b - a) * xi / 2 + (a + b) / 2
        y = (d - c) * zeta / 2 + (d + c) / 2
    else:
        # so far the other methods assume uniform grids 
        x = np.linspace(a, b, n + 1)
        y = np.linspace(c, d, m + 1)

    return np.meshgrid(x, y)
        
#Execution guard

if __name__ == "__main__":

    pass
