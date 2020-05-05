import matplotlib
import platform
if platform.system() == "linux":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import itertools
import dionysus as di
from geneo.utils import plot_diagram, is_included, get_order_of_magnitude
from geneo.constants import PERSISTENCE_PARAMS

class IENEO:
    """Generate n-dimensional non-expansive operators equivariant with respect
    to the group of isometries and built as a linear combination of symmetric
    Gaussians. It also manages basic operations as convolution and computation
    of the persistent diagrams of F(I)

    Parameters
    ----------
    size : int
        Size of the (square) kernel to be generated
    sigma : float, int, list
        standard deviation of the Gaussians to be combined. If float is
        interpreted as a fixed value. An int will be used to select randomly
        n standard deviations in [0,0.1] for the n Gaussians to be combined.
        Finally, a list specifying a standard deviation for each Gaussian will
        be simply be used as is.
    centers : int, list
        If int  the n number of centers will be chosen randomly in [-1,1]. A list
        will be used as is.

    Attributes
    ----------
    components : int
        number of 1-d Gaussians mixed to generate the kernel
    g : np.ndarray
        graph of the 1-d Gaussian
    """
    def __init__(self, size, sigma=None, centers=None, signal_dim=None):
        self.size = size
        self.centers = centers
        self.components = len(self.centers)
        self.sigma = sigma
        self.g = self.generate_1d()
        if signal_dim is not None:
            self.generate(dim =signal_dim)


    def update_parameters(self, coefficients, centers, sigma):
        """Allows to update the parameters and generate the kernel
        """
        self.centers = centers
        self.sigma = sigma
        self.g = self.generate_1d(coefficients = coefficients)
        self.generate(dim =self.dim)


    @property
    def number_of_parameters(self):
        if isinstnace(self.centers, list):
            c = len(self.centers)
        elif isinstance(self.centers, int):
            c = self.centers
        if isinstance(self.sigma, float):
            return 2 * c
        elif isinstance(self.sigma, int) or isinstance(self.sigma, list):
            return 3 * c


    @staticmethod
    def scale(array, target_range = (-1,1)):
        """Port values of arrayin target range
        """
        if not is_included(array, target_range):
            mini = array.min()
            maxi = array.max()
            l = target_range[0]
            u = target_range[1]
            array = (array - mini) / (maxi - mini)
            array = array * (u - l) + l
        return array


    @staticmethod
    def uniform_initializer(num_coeffs, target_range = (-1,1)):
        r = target_range
        return (r[1] - r[0]) * np.random.random_sample(num_coeffs) + r[0]


    @staticmethod
    def coeffs_normalization_(coeff, center):
        return coeff / center if center !=0 else coeff


    def coeffs_normalization(self, regularize = True):
        """Generating the filters by revolving 1d arrays around the origin,
        it is possible to normalize the coefficients to compoensate for the
        difference in area spanned by each Gaussian as a function of its
        distance from the origin"""
        coeffs = [self.coeffs_normalization_(co, ce)
                  for co, ce in zip(self.coefficients, self.centers)]
        if regularize:
            return coeffs / np.linalg.norm(np.asarray(coeffs))**2
        else:
            return coeffs


    @property
    def centers(self):
        return self._centers


    @centers.setter
    def centers(self, new_centers):
        if isinstance(new_centers, int):
            self._centers = self.uniform_initializer(new_centers,
                                                    target_range = (-1,1))
        else:
            new_centers = np.asarray(new_centers)
            assert new_centers.ndim == 1, "The array of centers should have dimension 1"
            self._centers = self.scale(new_centers)


    @property
    def sigma(self):
        return self._sigma


    @sigma.setter
    def sigma(self, new_sigma):
        if isinstance(new_sigma, float):
            self._sigma = [new_sigma for c in range(self.components)]
        elif isinstance(new_sigma, int):
            self._sigma = self.uniform_initializer(new_sigma,
                                                   target_range = (0,.1))
        else:
            self._sigma = new_sigma
        assert len(self.sigma) == len(self.centers), "Provide as many values for sigma as centers"


    @staticmethod
    def init_kernel_tensor(size, dim):
        dimensions = [size for i in range(dim)]
        return np.zeros(dimensions)


    def generate(self, dim = 2):
        """Generate an operator as a normalized linear combination of symmetric
        Gaussian
        """
        self.dim = dim
        if not hasattr(self, 'g'):
            self.g = self.generate_1d()
        x = np.linspace(-.5, .5, self.size)
        grid = itertools.product(x, repeat = dim)
        indices = itertools.product(range(self.size), repeat = dim)
        self.kernel = self.init_kernel_tensor(self.size, dim)
        int_func = interp1d(x, self.g)

        for index, values in zip(indices, grid):
            self.kernel[index] = int_func(sum([v**2 for v in values]))

    @staticmethod
    def gaussian(eval, center, sigma):
        return np.exp(( -(eval - center)**2) / (2 * sigma**2))

    def get_sym_gaussian(self, x, c, s):
        return self.gaussian(x, c, s) + self.gaussian(-x, c, s)

    def generate_1d(self, coefficients = None):
        if coefficients is None:
            self.coefficients = self.uniform_initializer(self.components)
        else:
            self.coefficients = coefficients
        self.coefficients = self.coeffs_normalization(regularize = True)
        x = np.linspace(-1, 1, self.size)
        sym_gaussians = np.asarray([coeff * self.get_sym_gaussian(x, center, s)
                         for coeff, center, s
                         in zip(self.coefficients, self.centers, self.sigma)])
        sym_gaussians = np.sum(sym_gaussians, axis = 0)
        sym_gaussians += abs(sym_gaussians.min())
        return sym_gaussians

    # @classmethod
    # def update_parameters(cls, size, sigma, centers, input_signal_dim):
    #     self.size = size
    #     self.centers = centers
    #     self.components = len(self.centers)
    #     self.sigma = sigma
    #     self.g = self.generate_1d()
    #     self.generate(dim = input_signal_dim)
    #     return self.kernel

    def convolve(self, image, params = None):
        """Convolve the kernel with the image by using scipy.ndimage.convolve
        with the parameters specified in params
        """
        return (sp.ndimage.convolve(image, self.kernel) if params is None
                else sp.ndimage.convolve(image, self.kernel, **params))

    def convolve_and_get_persistence(self, image, convolve = True,
                                     convolve_params = None,
                                     persistence_params = None):
        """Convolve the kernel with image and computes the persistent homology
        of the convolved image, according to the parameters specified in
        `convolve_params` and `persistence_params`
        """
        convolved = self.convolve(image, convolve_params) if convolve else image
        filtered_image = di.fill_freudenthal(convolved.astype(np.float64),
                                    reverse = persistence_params['reverse'])
        persistence = di.homology_persistence(filtered_image,
                                              persistence_params['field'],
                                              persistence_params['method'],
                                              persistence_params['progress'])
        return di.init_diagrams(persistence, filtered_image)

    def get_distance_on_signal(self, other, signal, hom_deg, persistence_params,
                               convolve_params):
        """Convolves self and other with signal. Computes the persistence
        diagrams for each filter and returns and bottleneck distance of their
        hom_deg-th persistence diagrams
        """
        pds1 = self.convolve_and_get_persistence(signal,
                                          persistence_params=persistence_params,
                                          convolve_params=convolve_params)
        pds2 = other.convolve_and_get_persistence(signal,
                                          persistence_params=persistence_params,
                                          convolve_params=convolve_params)
        return di.bottleneck_distance(pds1[hom_deg], pds2[hom_deg])

    def get_signals_distance(self, signal1, signal2, hom_deg, persistence_params,
                             convolve_params):
        pds1 = self.convolve_and_get_persistence(signal1,
                                          persistence_params=persistence_params,
                                          convolve_params=convolve_params)
        pds2 = self.convolve_and_get_persistence(signal2,
                                          persistence_params=persistence_params,
                                          convolve_params=convolve_params)
        return di.bottleneck_distance(pds1[hom_deg], pds2[hom_deg])


    def visualise_conv(self, image, convolve_params = None,
                       persistence_params = None, axes = None, hom_deg = 1):
        """Visualise the 1 and 2-dimensional rendering of the kernel, an image,
        the result of the convolution and the corresponding persistent diagram
        of degree `hom_deg`
        """
        if axes is None:
            fig, axs = plt.subplots(2,3)
            axs = axs.ravel()
        if persistence_params is None:
            persistence_params = PERSISTENCE_PARAMS
        convolved_image = self.convolve(image, params = convolve_params)
        dgms = self.convolve_and_get_persistence(convolved_image,
                                                 convolve = False,
                                                 persistence_params =
                                                 persistence_params)
        axs.ravel()
        axs[0].plot(self.g)
        axs[1].imshow(self.kernel)
        axs[2].imshow(image)
        axs[3].imshow(convolved_image)
        plot_diagram(dgms[hom_deg], ax = axs[4])

    def __repr__(self):
        return ("Isometry-equivariant operator"+
        " of shape {},\n".format([self.size for i in range(self.dim)]) +
        "mixture of {} symmetric gaussians\n".format(self.components) +
        "centered in {}\nand with sigmas {}\n".format(self.centers, self.sigma))

    def get_params(self):
        return np.concatenate((self.centers, self.sigma, self.coefficients),
                              axis=0)






class GIENEO:
    """Generate n-dimensional non-expansive operators equivariant with respect
    to the group of isometries and built as a linear combination of symmetric
    Gaussians. It also manages basic operations as convolution and computation
    of the persistent diagrams of F(I)

    Parameters
    ----------
    size : int
        Size of the (square) kernel to be generated
    sigma : float, int, list
        standard deviation of the Gaussians to be combined. If float is
        interpreted as a fixed value. An int will be used to select randomly
        n standard deviations in [0,0.1] for the n Gaussians to be combined.
        Finally, a list specifying a standard deviation for each Gaussian will
        be simply be used as is.
    centers : int, list
        If int  the n number of centers will be chosen randomly in [-1,1]. A list
        will be used as is.

    Attributes
    ----------
    components : int
        number of 1-d Gaussians mixed to generate the kernel
    g : np.ndarray
        graph of the 1-d Gaussian
    """
    def __init__(self, size, sigma=None, centers=None, signal_dim=None):
        self.size = size
        self.centers = centers
        self.components = len(self.centers)
        self.sigma = sigma
        self.g = self.generate_1d()
        if signal_dim is not None:
            self.generate(dim =signal_dim)


    def update_parameters(self, coefficients, centers, sigma):
        """Allows to update the parameters and generate the kernel
        """
        self.centers = centers
        self.sigma = sigma
        self.g = self.generate_1d(coefficients = coefficients)
        self.generate(dim =self.dim)


    @property
    def number_of_parameters(self):
        if isinstnace(self.centers, list):
            c = len(self.centers)
        elif isinstance(self.centers, int):
            c = self.centers
        if isinstance(self.sigma, float):
            return 2 * c
        elif isinstance(self.sigma, int) or isinstance(self.sigma, list):
            return 3 * c


    @staticmethod
    def scale(array, target_range = (-1,1)):
        """Port values of arrayin target range
        """
        if not is_included(array, target_range):
            mini = array.min()
            maxi = array.max()
            l = target_range[0]
            u = target_range[1]
            array = (array - mini) / (maxi - mini)
            array = array * (u - l) + l
        return array


    @staticmethod
    def uniform_initializer(num_coeffs, target_range = (-1,1)):
        r = target_range
        return (r[1] - r[0]) * np.random.random_sample(num_coeffs) + r[0]


    @staticmethod
    def coeffs_normalization_(coeff, center):
        return coeff / center if center !=0 else coeff


    def coeffs_normalization(self, regularize = True):
        """Generating the filters by revolving 1d arrays around the origin,
        it is possible to normalize the coefficients to compoensate for the
        difference in area spanned by each Gaussian as a function of its
        distance from the origin"""
        coeffs = [self.coeffs_normalization_(co, ce)
                  for co, ce in zip(self.coefficients, self.centers)]
        if regularize:
            return coeffs / np.linalg.norm(np.asarray(coeffs))**2
        else:
            return coeffs


    @property
    def centers(self):
        return self._centers


    @centers.setter
    def centers(self, new_centers):
        if isinstance(new_centers, int):
            self._centers = self.uniform_initializer(new_centers,
                                                    target_range = (-1,1))
        else:
            new_centers = np.asarray(new_centers)
            assert new_centers.ndim == 1, "The array of centers should have dimension 1"
            self._centers = self.scale(new_centers)


    @property
    def sigma(self):
        return self._sigma


    @sigma.setter
    def sigma(self, new_sigma):
        if isinstance(new_sigma, float):
            self._sigma = [new_sigma for c in range(self.components)]
        elif isinstance(new_sigma, int):
            self._sigma = self.uniform_initializer(new_sigma,
                                                   target_range = (0,.1))
        else:
            self._sigma = new_sigma
        assert len(self.sigma) == len(self.centers), "Provide as many values for sigma as centers"


    @staticmethod
    def init_kernel_tensor(size, dim):
        dimensions = [size for i in range(dim)]
        return np.zeros(dimensions)


    def generate(self, dim = 2):
        """Generate an operator as a normalized linear combination of symmetric
        Gaussian
        """
        self.dim = dim
        if not hasattr(self, 'g'):
            self.g = self.generate_1d()
        x = np.linspace(-.5, .5, self.size)
        grid = itertools.product(x, repeat = dim)
        indices = itertools.product(range(self.size), repeat = dim)
        self.kernel = self.init_kernel_tensor(self.size, dim)
        int_func = interp1d(x, self.g)

        for index, values in zip(indices, grid):
            self.kernel[index] = int_func(sum([v**2 for v in values]))

    @staticmethod
    def gaussian(eval, center, sigma):
        return np.exp(( -(eval - center)**2) / (2 * sigma**2))

    def get_sym_gaussian(self, x, c, s):
        return self.gaussian(x, c, s) + self.gaussian(-x, c, s)

    def generate_1d(self, coefficients = None):
        if coefficients is None:
            self.coefficients = self.uniform_initializer(self.components)
        else:
            self.coefficients = coefficients
        self.coefficients = self.coeffs_normalization(regularize = True)
        x = np.linspace(-1, 1, self.size)
        sym_gaussians = np.asarray([coeff * self.get_sym_gaussian(x, center, s)
                         for coeff, center, s
                         in zip(self.coefficients, self.centers, self.sigma)])
        sym_gaussians = np.sum(sym_gaussians, axis = 0)
        sym_gaussians += abs(sym_gaussians.min())
        return sym_gaussians

    # @classmethod
    # def update_parameters(cls, size, sigma, centers, input_signal_dim):
    #     self.size = size
    #     self.centers = centers
    #     self.components = len(self.centers)
    #     self.sigma = sigma
    #     self.g = self.generate_1d()
    #     self.generate(dim = input_signal_dim)
    #     return self.kernel

    def convolve(self, image, params = None):
        """Convolve the kernel with the image by using scipy.ndimage.convolve
        with the parameters specified in params
        """
        return (sp.ndimage.convolve(image, self.kernel) if params is None
                else sp.ndimage.convolve(image, self.kernel, **params))

    def convolve_and_get_persistence(self, image, convolve = True,
                                     convolve_params = None,
                                     persistence_params = None):
        """Convolve the kernel with image and computes the persistent homology
        of the convolved image, according to the parameters specified in
        `convolve_params` and `persistence_params`
        """
        convolved = self.convolve(image, convolve_params) if convolve else image
        filtered_image = di.fill_freudenthal(convolved.astype(np.float64),
                                    reverse = persistence_params['reverse'])
        persistence = di.homology_persistence(filtered_image,
                                              persistence_params['field'],
                                              persistence_params['method'],
                                              persistence_params['progress'])
        return di.init_diagrams(persistence, filtered_image)

    def get_distance_on_signal(self, other, signal, hom_deg, persistence_params,
                               convolve_params):
        """Convolves self and other with signal. Computes the persistence
        diagrams for each filter and returns and bottleneck distance of their
        hom_deg-th persistence diagrams
        """
        pds1 = self.convolve_and_get_persistence(signal,
                                          persistence_params=persistence_params,
                                          convolve_params=convolve_params)
        pds2 = other.convolve_and_get_persistence(signal,
                                          persistence_params=persistence_params,
                                          convolve_params=convolve_params)
        return di.bottleneck_distance(pds1[hom_deg], pds2[hom_deg])

    def get_signals_distance(self, signal1, signal2, hom_deg, persistence_params,
                             convolve_params):
        pds1 = self.convolve_and_get_persistence(signal1,
                                          persistence_params=persistence_params,
                                          convolve_params=convolve_params)
        pds2 = self.convolve_and_get_persistence(signal2,
                                          persistence_params=persistence_params,
                                          convolve_params=convolve_params)
        return di.bottleneck_distance(pds1[hom_deg], pds2[hom_deg])


    def visualise_conv(self, image, convolve_params = None,
                       persistence_params = None, axes = None, hom_deg = 1):
        """Visualise the 1 and 2-dimensional rendering of the kernel, an image,
        the result of the convolution and the corresponding persistent diagram
        of degree `hom_deg`
        """
        if axes is None:
            fig, axs = plt.subplots(2,3)
            axs = axs.ravel()
        if persistence_params is None:
            persistence_params = PERSISTENCE_PARAMS
        convolved_image = self.convolve(image, params = convolve_params)
        dgms = self.convolve_and_get_persistence(convolved_image,
                                                 convolve = False,
                                                 persistence_params =
                                                 persistence_params)
        axs.ravel()
        axs[0].plot(self.g)
        axs[1].imshow(self.kernel)
        axs[2].imshow(image)
        axs[3].imshow(convolved_image)
        plot_diagram(dgms[hom_deg], ax = axs[4])

    def __repr__(self):
        return ("Isometry-equivariant operator"+
        " of shape {},\n".format([self.size for i in range(self.dim)]) +
        "mixture of {} symmetric gaussians\n".format(self.components) +
        "centered in {}\nand with sigmas {}\n".format(self.centers, self.sigma))

    def get_params(self):
        return np.concatenate((self.centers, self.sigma, self.coefficients),
                              axis=0)



if __name__ == "__main__":
    plt.ion()
    print("0")
    g = IENEO(size = 27, sigma = 5, centers = 5)
    print("1")
    fig, ax_arr = plt.subplots(1,2)
    input()
    #print("1.5")
    ax_arr[0].plot(g.g)
    print("2")
    g.generate()
    input()
    ax_arr[1].imshow(g.kernel)
    input()
    plt.show()
