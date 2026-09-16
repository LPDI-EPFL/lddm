# Standard imports
import math
import numpy as np
from scipy.special import iv

# Adapted from https://github.com/engelen/vonmiseskde/blob/master/vonmiseskde/__init__.py

def von_mises_pdf(x: list[float],
                  mu: float = 0.0,
                  kappa: float = 1.0):
    
    num = np.exp(kappa * np.cos(x - mu))
    den = 2 * np.pi * iv(0, kappa)

    return num / den


def normalize_angles(angles: list[float]):
    """ Normalize a list of angles (in radians) to the range [-pi, pi]
    
    @param List[float] Input angles (in radians)
    @return List[float] Normalized angles (in radians)
    """
    # Change range to 0 to 2 pi
    angles = np.array(angles % (np.pi * 2))

    # Change range to -1 pi to 1 pi
    angles[angles > np.pi] = angles[angles > np.pi] - np.pi * 2
    
    return angles


class VonMisesKDE():
    
    def __init__(self, 
                 bandwidth=1.0):
        self.bandwidth = bandwidth
    
    def fit(self,
            X: np.array,):
        """
        X: angles in radians
        """
        
        assert len(X.shape) > 1, "Input data must be 2D to match sklearn interface, even though we only deal with 1D data."
        
        self.X = X
        self.normalized_X = normalize_angles(X.squeeze())

        self.kde_sample_X = np.linspace(-math.pi, math.pi, 1000)

        # Kernels, centered at input data points
        kernels = []

        for datapoint in self.X:
            # Make the basis function as a von mises PDF
            kernel = von_mises_pdf(self.kde_sample_X, 
                                    mu=datapoint,
                                    kappa=self.bandwidth)
            kernels.append(kernel)
        
        # Normalize pdf
        vmkde = np.sum(kernels, axis=0)
        integral = np.trapz(vmkde, x=self.kde_sample_X)
        self.kde_sample_y = vmkde / integral
    
    def score_samples(self, 
                      X):
        """ Evaluate the KDE at some inputs points
        
        @param List[float] input_x Input points
        @param List[float] Probability densities at input points
        """
        
        assert len(X.shape) > 1, "Input data must be 2D to match sklearn interface, even though we only deal with 1D data."
        X = X.squeeze()
        
        # Normalize inputs
        X = normalize_angles(angles=X)
        likelihoods = np.interp(X, self.kde_sample_X, self.kde_sample_y) + 1e-10
        log_likelihoods = np.log(likelihoods)

        return log_likelihoods