from geneo.read_data import DataSet
from geneo.gaussian_ieneo import GIENEO
from geneo.observer import Observer
from geneo.utils import init_operators
from geneo.classifiers import Brute
import numpy as np
import matplotlib.pyplot as plt
import geneo.data_augmentation
from geneo.data_augmentation import augment
import logging
import os
from tqdm import tqdm


def logger_init():
    # create logger with 'spam_application'
    logger = logging.getLogger('augmented_metric_learning')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('augmented_metric_learning.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


if __name__ == "__main__":
        logger = logger_init()
        path = '../classifiers/saved_observers'
        subdirs = [os.path.join(path, f) for f in os.listdir(path)]
        prepr_params = {'bw': False, 'threshold': .66, 'reshape': True,
                        'image_target': (128, 128), 'blur': True,
                        'kernel_size': (3,3),
                        'standardize': True}

        for subdir in tqdm(subdirs):

            for i, dataset in enumerate(tqdm(os.listdir(subdir))):
                logger.info("Working on dataset {}".format(dataset))

                for observer_path in tqdm(os.listdir(os.path.join(subdir, dataset))):
                    logger.info("Observer {}".format(observer_path))
                    class_num = observer_path.split("_")[-2:]
                    class_num = [int(class_num[0]), int(class_num[1][0])]
                    if 'fashion' not in dataset:
                        kernel_size = observer_path.split("_")[1]
                    else:
                        kernel_size = observer_path.split("_")[2]

                    kernel_shape_layer1 = (int(kernel_size),int(kernel_size))
                    if i == 0:
                        logger.info("initialize data")
                        data = DataSet(dataset,
                                       num_samples_from_training = None,
                                       num_classes = class_num)
                        data.preprocess(params_dict = prepr_params)
                        num_examples = 20
                        data.select_n_examples_per_class(num_examples=num_examples)
                    else:
                        logger.info("data already initialised")
                    observer = Observer.load(os.path.join(subdir, dataset,observer_path))
                    model = Brute(observer, data)
                    num_val_ex = 10
                    test = np.concatenate([model.data.sampled_val_imgs[c][:num_val_ex]
                                           for c in model.data.sampled_val_imgs], axis = 0)
                    rev_class_map = {data.class_map[k]: k for k in data.class_map}
                    test_labels = np.concatenate([np.ones(num_val_ex) * rev_class_map[c]
                                   for c in model.data.sampled_val_imgs], axis = 0).astype("int")

                    aug_test = augment(test)
                    path = dataset + "_" +str(kernel_size) + "_" + str(class_num) +"_augmented_test.npy"
                    np.save(path, aug_test)
                    logger.info("validate")
                    model.validate(aug_test, test_labels, func=np.max)
                    model.plot_validation_distance_matrix()
                    plt.savefig(dataset + "_" +str(kernel_size) + "_" + str(class_num) +"_augmented_matrix.svg")
                    model.plot_validation_dendrogram(method = "single", num_ex_per_class = num_val_ex)
                    plt.savefig(dataset + "_" +str(kernel_size) + "_" + str(class_num) +"_augmented_dendro.svg")
                    path = dataset + "_" +str(kernel_size) + "_" + str(class_num) +"_aug_test.npy"
                    np.save(path, aug_test)
