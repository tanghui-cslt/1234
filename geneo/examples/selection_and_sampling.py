from geneo.read_data import DataSet
from geneo.gaussian_ieneo import GIENEO
from geneo.observer import Observer
from geneo.utils import init_operators

print("initialize data")
data = DataSet('mnist', num_samples_from_training = None, num_classes =2)
print("preprocess")
preprocessing_dict = {'bw': False, 'threshold': .66,
                      'reshape': True, 'image_target': (128, 128),
                      'blur': True, 'kernel_size': (3,3),
                      'standardize': True}
data.preprocess(params_dict = preprocessing_dict)
num_examples =4
data.select_n_examples_per_class(num_examples=num_examples)
print("initialize operators")
operators = init_operators(GIENEO,
                          number =20,
                          size = 11,
                          sigma = .1,
                          centers = 5)
print("initialize oberver")
observer = Observer(operators)
hom_deg = 1
observer.select_filters_on_signal_set(data.sampled_imgs,
                                                  hom_deg = hom_deg,
                                                  threshold =1.5,
                                                  max_ops_per_class = 10)

print("\n{} operators have been selected".format(len(observer.selected_operators)))
observer.sample_operators(data.sampled_imgs, th_per = 75)
print("\n{} operators have been sampled".format(len(observer.sampled_operators)))
