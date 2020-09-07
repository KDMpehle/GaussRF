"""Generate point process fields"""
import warnings, inspect
import numpy as np
from scipy.optimize import fmin_tnc

class PoissHF_2D(object):
    """ The clas for a two-dimensional homogeneous Poisson process

    The field is instantiated with domain end points
    lims = [a, b, c, d] = [a, b] x [c, d]
    and intensity lamb (a positive number)
    """

    def __init__(self, lims, lamb):
        #instantiate the 2D Poisson field
        self.lims = lims
        self.a, self.b = lims[0:2] #get the horizontal limits
        self.c, self.d = lims[2:4] #get the vertical limits
        self.A = (self.b - self.a) * (self.d - self.c) #area of the rectangular domain
        self.lamb = lamb

    def __str__(self):
        #The str method
        return (
            "2D Poisson homogeneous random field on ["
            + str(self.a)
            + ", "
            + str(self.b)
            + "] x ["
            + str(self.c)
            + ", "
            + str(self.d)
            + "] with intensity "
            + str(self.lamb)
            )

    def __repr__(self):
        #The repr method
        return (
            "PoissHF_2D(lims = "
            + str(self.lims)
            + ", lamb = "
            + str(self.lamb)
            + ")"
            )
    
    #Variable setters and getters
    @property
    def lims(self):
        #get the domain limits
        return self._lims

    @lims.setter
    def lims(self, value):
        if not isinstance(value, list):
            raise TypeError("Limits must be provided as a list")
        elif len(value) != 4:
            raise TypeError("Must provide 4 numbers for the limits")
        elif not ( all( isinstance(x, (int,float)) for x in value) and not any(isinstance(x, bool) for x in value) ):
            raise ValueError("Interval end points must be real numbers")
        elif (value[1] <= value[0]) or (value[3] <= value[2]):
            raise ValueError("Error in limits, a < b and c < d for rectangle [a,b] x [c,d]")
        self._lims = value
        self.a, self.b = self.lims[0:2]
        self.c, self.d = self.lims[2:4]
        self.A = (self.b - self.a) * (self.d - self.c)

    @property
    def lamb(self):
        #Get the constant intensity
        return self._lamb

    @lamb.setter
    def lamb(self, value):
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)) or value <= 0:
            raise TypeError("Homogeneous intensity must be a positive real number")
        self._lamb = value

    #Methods of the class
    def realisation(self):
        #Generate the number of points in the process
        N = np.random.poisson( self.A * self.lamb) #Number of points
        points = np.random.random((N, 2)) #initialise the points
        
        #translate the points to specified domain
        points[:,0] = points[:,0] * (self.b - self.a) + self.a
        points[:,1] = points[:,1] * (self.d - self.c) + self.c

        return points
        
class PoissHF_disk(object):
    """The class for a two-dimensional homogeneous Poisson process on a circular disk

    The field is instantiated with disk radius R and intesnity lamb
    """

    def __init__(self, R, lamb):
        #instantiate the 2D Poisson field on the disk
        self.R = R #Get the domain radius
        self.lamb = lamb
        self.A = np.pi * self.R**2 #area of the domain

    #Do the str method
    def __str__(self):
        #The str method
        return (
            "2D Poisson random field on the disk of radius"
            + str(self.R)
            + " and intensity "
            +str(self.lamb)
            )

    def __repr__(self):
        #The repr method.
        return (
            "PoissHF_disk(R = "
            + str(self.R)
            + ", lamb = "
            + str(self.lamb)
            + ")" )
    
    #Variable getters and setters
    @property
    def R(self):
        #Get the radius of the disk
        return self._R

    @R.setter
    def R(self, value):
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)) or value <= 0:
            raise TypeError("Homogeneous intensity must be a positive real number")
        self._R = value
    
    @property
    def lamb(self):
        #Get the constant intensity
        return self._lamb

    @lamb.setter
    def lamb(self, value):
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)) or value <= 0:
            raise TypeError("Homogeneous intensity must be a positive real number")
        self._lamb = value
        
    #Methods
    def realisation(self):
        #Generate the number of points in the process
        N = np.random.poisson(self.A * self.lamb) # number of points
        polar_pts = np.random.random((N,2)) # generate uniform random variables to be manipulated
        points = np.zeros((N,2))

        #Transform to polar coordinates 0 < r < R, 0 < theta < 2pi
        polar_pts[:, 0] = self.R * np.sqrt(polar_pts[:,0])
        polar_pts[:, 1] = 2 * np.pi * polar_pts[:, 1]
        
        #Convert to Cartesian coordinates
        points[:, 0] = polar_pts[:,0] * np.cos(polar_pts[:, 1])
        points[:, 1] = polar_pts[:,0] * np.sin(polar_pts[:, 1])

        return points
class PoissHF_sphere(object):
    """ The class for an homogeneous Poisson process on the sphere

    The field is instantiated with sphere radius R and
    intensity lamb

    """

    def __init__(self, R, lamb):
        #Instantiate the Poisson field on the sphere
        self.R = R
        self.lamb = lamb
        self.S = 4 * np.pi * self.R**2

    #Do the str method
    def __str__(self):
        #The str method
        return (
            "Poisson random field on the sphere of radius"
            + str(self.R)
            + " and intensity "
            +str(self.lamb)
            )

    def __repr__(self):
        #The repr method.
        return (
            "PoissHF_sphere(R = "
            + str(self.R)
            + ", lamb = "
            + str(self.lamb)
            + ")" )
    
    #Variable getters and setters
    @property
    def R(self):
        #Get the radius of the disk
        return self._R

    @R.setter
    def R(self, value):
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)) or value <= 0:
            raise TypeError("Homogeneous intensity must be a positive real number")
        self._R = value
    
    @property
    def lamb(self):
        #Get the constant intensity
        return self._lamb

    @lamb.setter
    def lamb(self, value):
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)) or value <= 0:
            raise TypeError("Homogeneous intensity must be a positive real number")
        self._lamb = value

    #Methods
    def realisation(self):
        #Generate number of points in the process
        N = np.random.poisson(self.S * self.lamb) # number of points in realisation
        angles = np.random.random((N, 2)) # to be transformed to uniform (theta, phi)
        points = np.zeros((N,3)) #preallocate the cartesian coordinates array

        #Transform the uniform (0,1) points to angular coordinates
        angles[:, 0] = np.arccos(2 * angles[:, 0] - 1)
        angles[:, 1] = 2 * np.pi * angles[:, 1]

        #Convert to Cartesian coordinates
        points[:, 0] = self.R * np.sin(angles[:, 0]) * np.cos(angles[:, 1])
        points[:, 1] = self.R * np.sin(angles[:, 0]) * np.sin(angles[:, 1])
        points[:, 2] = self.R * np.cos(angles[:, 0])

        return points
class PoissF_2D(object):
    """The class for a two-dimensional inhomogeneous Poisson process

    The field is instantiated with domain end points
    lims = [a, b, c, d]= [a, b] x [c, d]
    and intensity function lamb
    """

    def __init__(self, lims, lamb, lamb_max = None):
        #instantiate the 2D Poisson field
        self.lims = lims 
        self.a, self.b = lims[0:2] # get the horizontal limits
        self.c, self.d = lims[2:4] # get the vertical limits
        self.A = (self.b - self.a) * (self.d - self.c) # area of the rectangular domain.
        self.lamb = lamb # get the intensity
        self.lamb_max = lamb_max # the threshold lambda value for the thinning process
        self._threshold = self.threshold
        self._isfunc = callable(self.lamb) #Boolean check on if lambda is a function

    #do the str method
    def __str__(self):
        #The str method
        return (
            "2D Poisson random field on ["
            + str(self.a)
            + ", "
            + str(self.b)
            + "] x ["
            + str(self.c)
            + ", "
            + str(self.d)
            + "], with intensity "
            +self.lamb.__name__
            )
    def __repr__(self):
        #The repr method.
        return (
            "PoissF_2D(lims = "
            + str(self.lims)
            + ", lamb = "
            + str(self.lamb.__name__)
            + ")" )
    #Variable setters and getters.
    @property
    def lims(self):
        #get the domain limits
        return self._lims

    @lims.setter
    def lims(self, value):
        if not isinstance(value, list):
            raise TypeError("Limits must be provided as a list")
        elif len(value) != 4:
            raise TypeError("Must provide 4 numbers for the limits")
        elif not ( all( isinstance(x, (int,float)) for x in value) and not any(isinstance(x, bool) for x in value) ):
            raise ValueError("Interval end points must be real numbers")
        elif (value[1] <= value[0]) or (value[3] <= value[2]):
            raise ValueError("Error in limits, a < b and c < d for rectangle [a,b] x [c,d]")
        self._lims = value
        self.a, self.b = self.lims[0:2]
        self.c, self.d = self.lims[2:4]
        self.A = (self.b - self.a) * (self.d - self.c)

    @property
    def lamb(self):
        #get the intensity function
        return self._lamb

    @lamb.setter
    def lamb(self, value):
        if not callable(value) and ( not (isinstance(value, (int, float)) and not isinstance(value, bool)) or value <= 0 ):
            raise TypeError("constant intensity must be a real positive number")
        elif len(inspect.getfullargspec(value)[0]) != 2:
            raise ValueError("Intensity must be a bivariate function")
        self._lamb = value
        self._isfunc = callable(value) # see if intensity is a function

    @property
    def lamb_max(self):
        #get the threshold intensity
        return self._lamb_max

    @lamb_max.setter
    def lamb_max(self, value):
        if not (isinstance(value, (int,float)) and not isinstance(value,bool) and value>= 0) and value is not None:
            raise TypeError("Threshold homogeneous intensity must be a positive real number, if supplied.")
        self._lamb_max = value
    
    #Methods of the class
    def threshold(self):
        #This method gets the threshold intensity for the simulation procedure
        if self.lamb_max is not None:
            return self.lamb_max
        if self._isfunc == False:
            return self.lamb # in the case of an homogeneous Poisson process
        
        # For an inhomogeneous Poisson process
        # Approximate max value of the intensity function
        def func(x):
            return - self.lamb(x[0], x[1]) #negate function so scipy's fmin_tnc finds its maxima
        xmid, ymid = (self.b + self.a) / 2, (self.d + self.c) / 2
        x0 = np.array([xmid, ymid]) #guess for the maximum in the centre of the simulation domain
        boundary = [ (self.a, self.b), (self.c, self.d) ] # the boundary to input into fmin_tnc
        x_max = fmin_tnc(func, x0, approx_grad = True, bounds = boundary)[0] # return the point of maximum value
        return self.lamb(x_max[0], x_max[1]) # return threshold lambda_max for the thinning process

    def realisation(self):
        #Return a sample of the Poisson process: thinning procedure
        lamb_star = self._threshold() # get the thinning lambda rate
        print(lamb_star)
        N = np.random.poisson(lamb_star) # homogeneous Poisson number of points
        # Generate homogeneous Poisson of lambda_star on [a,b] x [c,d]
        x_pts = np.random.random((N, 2))
        x_pts[:,0] = self.a + (self.b - self.a) * x_pts[:,0]
        x_pts[:,1] = self.c + (self.d - self.c) * x_pts[:,1]
        # Thin the process
        indices = np.where(np.random.random(N) < self.lamb(x_pts[:,0], x_pts[:,1])/lamb_star)[0]
        return x_pts[indices, :]
    
        #for k in range(self.samples):
        #    x_points = np.random.random(N[k], 2) # uniform
        #    x_points[:,0] = self.a + (self.b - self.a) * x_points[:,0] # translate x-coordiantes from (0,1) to (a,b)
        #    x_points[:,1] = self.c + (self.d - self.c) * x_points[:,1] # translate y-coordinates from (0,1) to (c,d)
        #    indices = np.where( np.random.random(N[k]) < self.lamb(x_points[:,0], x_points[:,1]))/lamb_star)[0]
        #    points_sample[k] = x_points[indices,:] # added the thinned process to the sample
        #return points_sample # have to test this out in a simpler script

#Execution guard

if __name__ == "__main__":

    pass
#Do a test
#def lamb(x,y):
#    return 300 * ( x**2 + y**2 )
#lims = [0., 1., 0., 1.]
#points = PoissF_2D(lims, lamb).Poi_field()

#import matplotlib.pyplot as plt
#from matplotlib import cm
#plt.plot(points[:,0], points[:,1], 'o')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title(r'Poisson process with intensity $\lambda(x,y)$')
# make a heat map of the intensity function
#xx, yy = np.meshgrid(np.linspace(0., 1., 100), np.linspace(0., 1., 100))
#L = lamb(xx, yy)
#plt.pcolor(xx, yy, L, cmap = cm.gray)
#plt.colorbar()
#plt.savefig('Poi_inhomog_eg.pdf')
#plt.show()
