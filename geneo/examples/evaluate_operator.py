from geneo.read_data import DataSet
from geneo.gaussian_ieneo import IENEO
import matplotlib.pyplot as plt
from geneo.data_augmentation import random_rotation

print("initialize data")
data = DataSet('mnist', num_samples_from_training = 1)
print("preprocess")
preprocessing_dict = {'bw': True, 'threshold': .66,
                      'reshape': True, 'image_target': (256,256),
                      'blur': True, 'kernel_size': (3,3),
                      'standardize': True}
data.preprocess(params_dict = preprocessing_dict)
im = data.x_train[0]
g = IENEO(size = 11, sigma = .1, centers = 5)
g.generate()
g.visualise_conv(im)
im_ = random_rotation(im)
g.visualise_conv(im_)
im__ = random_rotation(im)
g.visualise_conv(im__)
plt.show()
