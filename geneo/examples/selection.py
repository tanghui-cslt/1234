from geneo.read_data import DataSet
from geneo.observer import Observer
from geneo.utils import init_operators
from geneo.gaussian_ieneo import GIENEO

print("initialize data")
data = DataSet('mnist', num_samples_from_training = None, num_classes =2)
print("preprocess")
preprocessing_dict = {'bw': False, 'threshold': .66, 'reshape': True,
                      'image_target': (128, 128), 'blur': True,
                      'kernel_size': (3,3), 'standardize': True}
data.preprocess(params_dict = preprocessing_dict)
num_examples =4
data.select_n_examples_per_class(num_examples=num_examples)
print("initialize operators")
operators = init_operators(GIENEO, number =5, size = 11, sigma = .1, centers = 5)
print("initialize oberver")
observer = Observer(operators)
observer.select_filters_on_signal_set(data.sampled_imgs, hom_deg = 1,
                                      threshold =1.5, max_ops_per_class = 10)
