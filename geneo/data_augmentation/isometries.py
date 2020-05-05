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

transforms = [random_rotation, random_translation, reflect_random_axis]
parameters = [ROTATION_PARAMS, TRANSLATION_PARAMS, {}]
