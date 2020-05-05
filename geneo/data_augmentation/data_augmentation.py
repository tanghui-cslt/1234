import numpy as np
import imutils
# from isometries import random_rotation, random_translation, reflect_random_axis
import numpy as np
import imutils
from geneo.constants import ROTATION_PARAMS, TRANSLATION_PARAMS

def random_rotation(image, range_theta = [15, 45]):
    """rotate an image of k degree, where k is chosen randomly in range_theta
    """
    angle = np.random.randint(range_theta[0], range_theta[1])
    return imutils.rotate(image, angle)

def random_translation(image, rangex = [0, 2], rangey = [0, 2], scale = [1,1]):
    """Translate an image in 2d of an amount of pixels randomly selected in
    rangex x rangey and scaled of [scalex, scaley] in case the image has been
    resized.
    """
    t_x = np.random.randint(rangex[0], rangex[1])
    t_y = np.random.randint(rangey[0], rangey[1])
    return imutils.translate(image, t_x*scale[0], t_y*scale[1])

def reflect_random_axis(image):
    """Generate a random reflection aroung one of the two orthogonal axes
    centered in the centre of the image
    """
    ax = np.random.randint(0, image.ndim - 1)
    return np.flip(image, ax)


def augment_n_times(function, inputs_dict, n):
    """Generate n augmented images by applying the same functions according to
    the parameters specified in inputs_dict
    """
    return [function(**inputs_dict) for i in range(n)]

def true_k_over_n_times(k = 1, n = 10):
    """Returns true k times over n drawing from a normal distribution
    """
    return np.random.randint(0, n) in range(k)

def augment_image(image, functions = None, parameters = None, probability = 10):
    """Augment an image given a specific transformation function. The
    transformation is applied randomly according to the percentage specifried in
    probability

    Parameters
    ----------
    image : np.ndarray
        a two dimensional matrix. We assume to work with grayscale images
    functions : func
        a python function
    parameters : dict
        dictionary of keyword arguments for func
    probability : int
        probability in percentage of transforming the image

    Returns
    -------
    type
        Description of returned object.

    """

    for f, p in zip(functions, parameters):
        if true_k_over_n_times(probability, 100):
            image = f(image, **p)

    return image

def augment(images):
    functions = [random_rotation, random_translation,
                 reflect_random_axis]
    rotation = {'range_theta':[1, 30]}
    translation = {'rangex':[1, 2], 'rangey':[1, 2]}
    reflection = {}
    parameters = [rotation, translation, reflection]
    return np.asarray([augment_image(image, functions = functions,
                                          parameters = parameters,
                                          probability = 50)
                       for image in images])
