from geneo.read_data import DataSet
from geneo.gaussian_ieneo import IENEO
from geneo.observer import Observer
from geneo.utils import init_operators
from geneo.classifiers import Brute
import numpy as np
import matplotlib.pyplot as plt
import logging

def logger_init():
    # create logger with 'spam_application'
    logger = logging.getLogger('representation')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('representation.log')
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
    logger.info("preprocess")
    prepr_params = {'bw': False, 'threshold': .66, 'reshape': True,
                    'image_target': (128, 128), 'blur': True,
                    'kernel_size': (3,3),
                    'standardize': True}

    datasets = ["cifar10", "mnist", "fashion_mnist"]

    for dataset in datasets:
        logger.info("Dataset {}".format(dataset))
        logger.info("initialize data")
        data = DataSet(dataset, num_samples_from_training = None, num_classes =2)
        data.preprocess(params_dict = prepr_params)

        logger.info("Select training examples")
        num_examples = 20
        data.select_n_examples_per_class(num_examples=num_examples)
        sizes = [7,11,21]

        for size in sizes:
            logger.info("size {}".format(size))
            try:
                logger.info("initialize operators")
                operators = init_operators(GIENEO, number=750, size=size, sigma=.1,
                                           centers = int(size/2)+1)
                logger.info("initialize observer and model")
                model = Brute(Observer(operators), data)
                logger.info("fit")
                model.fit(hom_deg=1, num_examples=num_examples, threshold=1.5,
                          th_per=75, select=True, sample=True)
                logger.info("generate test from a set of train examples")
                num_train_ex = 10
                train = np.concatenate([model.data.sampled_imgs[c][:num_train_ex]
                                       for c in model.data.sampled_imgs], axis = 0)
                rev_class_map = {data.class_map[k]: k for k in data.class_map}
                train_labels = np.concatenate([np.ones(num_train_ex) * rev_class_map[c]
                               for c in model.data.sampled_imgs], axis = 0).astype("int")
                logger.info("validate")
                model.validate(train, train_labels, func=np.max)
                model.plot_validation_distance_matrix()
                plt.savefig(dataset + "_" +str(size) +"_train_matrix.svg")
                model.plot_validation_dendrogram(method = "single")
                plt.savefig(dataset + "_" +str(size) +"_train_dendro.svg")
                logger.info("generate test from a set of validation examples")
                num_val_ex = 10
                test = np.concatenate([model.data.sampled_val_imgs[c][:num_val_ex]
                                       for c in model.data.sampled_val_imgs], axis = 0)
                rev_class_map = {data.class_map[k]: k for k in data.class_map}
                test_labels = np.concatenate([np.ones(num_val_ex) * rev_class_map[c]
                               for c in model.data.sampled_val_imgs], axis = 0).astype("int")
                logger.info("validate")
                model.validate(test, test_labels, func=np.max)
                model.plot_validation_distance_matrix()
                plt.savefig(dataset + "_" +str(size) +"_matrix.svg")
                model.plot_validation_dendrogram(method = "single")
                plt.savefig(dataset + "_" +str(size) +"_dendro.svg")
                path = dataset + "_" +str(size) + ".npy"
                model.observer.save(path)
            except Exception as e:
                logger.info("Size {} failed for dataset {} with error {}".format(size, dataset, e))
