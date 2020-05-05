import importlib as imp
import platform
import numpy as np
if platform.system() == "linux":
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import cv2 as cv
from tqdm import tqdm
from geneo.constants import AVAILABLE_DATASETS, PREPROC_PARAMS
import geneo.data_augmentation as daug
import dionysus as di
import keras

class DataSet:
    """Gives standard datasets for supervised machine learning tasks by using
    keras.datasets. Gives the possibility to preprocess and plot the images.

    Parameters
    ----------
    name : str
        A standard dataset among
        ["mnist", "fashion_mnist", "cifar10", "cifar100"].
    label_mode : str
        Specific attribute for CIFAR100 labelling taken directly from keras.
    """
    def __init__(self, name = None, label_mode = 'fine',
                 num_samples_from_training= None, num_classes = None):
        self.name = name
        if self.name is 'cifar100':
            (self.x_train, self.y_train),\
            (self.x_test, self.y_test) = self.data_set.load_data(label_mode=label_mode)
        else:
            (self.x_train, self.y_train),\
            (self.x_test, self.y_test) = self.data_set.load_data()
        if self.x_train.shape[-1] == 3:
            print("Found 3 channels, this implementation only supports grayscale")
            self.x_train = self.x_train[:,:,:,0]
            self.x_test = self.x_test[:,:,:,0]
        self.classes = np.unique(self.y_train)
        #print(self.classes)
        if num_classes is not None:
            self.limit_to_n_classes(num_classes)
        if num_samples_from_training is not None:
            perm = np.random.permutation(len(self.x_train))
            self.x_train = self.x_train[perm][:num_samples_from_training]
            self.y_train = self.y_train[perm][:num_samples_from_training]
        self.y_train = keras.utils.np_utils.to_categorical(self.y_train)
        self.y_test = keras.utils.np_utils.to_categorical(self.y_test)


    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        if new_name in AVAILABLE_DATASETS:
            self._name = new_name
            self.data_set = imp.import_module("keras.datasets.{}".format(new_name))
        else:
            raise ValueError("Please select one of" +
                             " the dataset available" +
                             " in {}".format(AVAILABLE_DATASETS))

    def limit_to_n_classes(self, n):
        """Select n random classes from the dataset (if the requested n is less
        than the number of available classes)
        """
        if isinstance(n, int):
            print("-"*50, "Select {} classes randomly".format(n))
            if n < len(self.classes):
                p = np.random.permutation(len(self.classes))
            elif n == len(self.classes):
                p = np.arange(n)
            self.classes = self.classes[p][:n]
        elif isinstance(n, list):
            print("-"*50, "Training on classes number {} ".format(n))
            self.classes = self.classes[n]

        indices = np.where(np.isin(self.y_train, self.classes))[0]
        self.x_train = self.x_train[indices]
        self.y_train = self.y_train[indices]
        indices = np.where(np.isin(self.y_test, self.classes))[0]
        self.x_test = self.x_test[indices]
        self.y_test = self.y_test[indices]
        self.update_labels()
        if isinstance(n, int):
            self.classes = np.array(range(n))
        else:
            self.classes = np.array(range(len(n)))


    def update_labels(self):
        """If a subset of labels is selected updates the labels to make the
        one hot encoding possible. Creates self.class_map where the map between
        the old and new values of the labels is stored
        """
        self.class_map = {c : i for i,c in enumerate(self.classes)}

        for c in self.class_map:
            self.y_train[self.y_train == c] = self.class_map[c]
            self.y_test[self.y_test == c] = self.class_map[c]

    @staticmethod
    def preprocess_(img,
                    bw = True, threshold = .66,
                    reshape = True, image_target = (256,256),
                    blur = True, kernel_size=(5,5),
                    standardize=True):
        """preprocess a single image"""
        if bw:
            img[img < 255*threshold] = 0
        if reshape:
            img = cv.resize(img,image_target, interpolation = cv.INTER_CUBIC)
        if blur:
            img = cv.GaussianBlur(img, kernel_size, 0)
        if standardize:
            img = (img - np.mean(img)) / np.std(img)
            img = ((img - img.min()) / (img.max() - img.min()))*255
            return img.astype('uint8')
        else:
            return img

    def preprocess(self, func = None, params_dict = None):
        """Preprocesses the train and test image sets by applying the function
        func[params_dict] to each image. If func is None the function
        self.preprocess_ is used, if params_dict is None,
        :const:`geneo.constants.PREPROC_PARAMS` is used.
        """
        if func is None:
            func = self.preprocess_
        if params_dict is None:
            params_dict = PREPROC_PARAMS
        if params_dict['reshape']:
            target_shape = np.array(params_dict['image_target'])
            original_shape = np.array(self.x_train[0].shape[:2])
            self.res_ratio = target_shape / original_shape
        else:
            self.res_ratio = [1,1]
        prep_train = []
        prep_test = []

        for i, img in enumerate(tqdm(self.x_train, desc="preprocessing train")):
            prep_train.append(func(img, **params_dict))

        self.x_train = np.asarray(prep_train)

        for i, img in enumerate(tqdm(self.x_test, desc="preprocessing test")):
            prep_test.append(func(img, **params_dict))

        self.x_test = np.asarray(prep_test)

    def select_n_examples_per_class(self, num_examples):
        """Selects n examples per class available in self.classes. Returns the
        minimum number of examples found. If available it
        """
        self.sampled_imgs = {c:[] for c in self.classes}
        self.sampled_val_imgs = {c:[] for c in self.classes}
        labels = self.one_hot2dense(self.y_train)
        true_num_examples = num_examples

        for c in self.sampled_imgs:
            imgs_c = [im for i, im in enumerate(self.x_train)
                                    if labels[i] == c ]
            self.sampled_imgs[c] = imgs_c[:num_examples]
            try:
                self.sampled_val_imgs[c] = imgs_c[num_examples : ]
            except:
                print("No validation examples found for class {}".format(c))
            if len(self.sampled_imgs[c]) < true_num_examples:
                true_num_examples = len(self.sampled_imgs[c])

        self.num_examples_per_class = true_num_examples

    @staticmethod
    def one_hot2dense(array):
        """Returns labels densely encoded
        """
        if isinstance(array[0], (list, tuple, np.ndarray)):
            labels = np.argmax(array, axis = 1)
        elif isinstance(array[0], int):
            labels = array
        else:
            raise ValueError("Labels (data.y_train) data type not understood")
        return labels

    def plot_image(self, image = None, index = None, cmap = 'gray'):
        """Plots the image store in self.x_train[index] if index is provided,
        otherwise plots a random image
        """
        fig, ax = plt.subplots()
        if image is None:
            if index is None:
                print("plotting random image")
                index = np.random.choice(self.x_train.shape[0], 1)
            image = self.x_train[index]
        try:
            ax.imshow(image, cmap = cmap)
        except:
            ax.imshow(np.squeeze(image), cmap = cmap)
        plt.show()

    def augment(self, substitute = True):
        """Performs data augmentation, so far only by applying 2D isometries.
        If substitute is True augmented images are substituted to the original
        in the training dataset, otherwise the augmented images will be appended
        at the end of the training set, as well as their labels
        """
        rot_params = {'range_theta': [15, 45]}
        translation_params = {'rangex': [0, 2],
                              'rangey': [0, 2],
                              'scale': self.res_ratio}
        ref_params = {}
        params_dicts = [rot_params,
                        translation_params,
                        ref_params]
        if not substitute:
            augmented_images = []
        description = "Augment data. Substitution set to {}".format(substitute)

        for i, img in enumerate(tqdm(self.x_train, desc = description)):
            image = daug.augment_image(img, functions = daug.iso_transforms,
                                       parameters = params_dicts)
            if substitute:
                self.x_train[i,] = image
            else:
                augmented_images.append(image)
                augmented_labels.append(self.y_train[i])

        if not substitute:
            augmented_images = np.asarray(augmented_images)
            self.x_train = np.concatenate((self.x_train, augmented_images),
                                          axis = 0)
            augmented_labels = np.array(augmented_labels)
            self.y_train = np.concatenate((self.y_train, augmented_labels),
                                          axis = 0)

    def __repr__(self):
        return ("Working with " +
            "{}, {} training samples".format(self.name, len(self.x_train)) +
            " and images of size {}.".format(str(self.x_train[0].shape)))


if __name__ == "__main__":
    d = DataSet('mnist', num_samples_from_training = 100)
    d.preprocess()
    d.augment()
    d.plot_image()
    plt.show()
