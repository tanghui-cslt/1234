import keras.backend as K
from keras.callbacks import EarlyStopping
import keras
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from geneo.read_data import DataSet
from geneo.observer import Observer
from geneo.utils import init_operators
from geneo.gaussian_ieneo import IENEO
from geneo.classifiers.utils import build_cnn, build_fc, plot_net_history
from geneo.constants import TRAIN_PARAMS
from geneo.classifiers import Brute
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
import logging

def logger_init():
    # create logger with 'spam_application'
    logger = logging.getLogger('knowledge_transfer')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('knowledge_transfer.log')
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


def geneo_init(shape = None, operators = None, dtype=None):
    init =np.zeros(shape)
    #print("init")
    #print(init)
    logger.info("from intializer {}".format(init.shape))
    if shape[-1] <= len(operators):
        operators = np.asarray([op.kernel for op in operators])
        operators = operators[np.random.permutation(len(operators))][:shape[-1]]
        operators = np.transpose(operators, axes = (2,1,0))
        init[:,:,0,:] = operators
        #print (init)
    else:

        for j in range(shape[-1]):
            op = np.random.choice(operators)
            init[:,:,0,j] = op.kernel

    return init

def build_net(arch = None, params = None, input_shape = None,
              num_classes = None):
    if arch == 'fc':
        return build_fc(input_shape, num_classes=num_classes,  **params)
    elif arch == 'cnn':
        assert isinstance(params, list),"provide [cnn_params,fc_params]"
        return build_cnn(input_shape, num_classes=num_classes,
                         cnn_dict = params[0], fc_dict = params[1])
    else:
        raise NotImplementedError("Architecture {} unknown.".format(arch))

def compile(net, train_parameters = None):
    net.compile(loss = train_parameters['loss'],
                optimizer = train_parameters['optimizer'](lr=train_parameters['lr']),
                metrics = train_parameters['metrics'])
    a = (None,128,128,1)
    net.build(a)
    net.summary()
    return net

if __name__ == "__main__":
    logger = logger_init()
    path = './train_all_data'
    subdirs = [os.path.join(path, f) for f in os.listdir(path)]
    prepr_params = {'bw': False, 'threshold': .66, 'reshape': True,
                    'image_target': (128, 128), 'blur': True,
                    'kernel_size': (3,3),
                    'standardize': True}

    for subdir in tqdm(subdirs):

        for i, dataset in enumerate(tqdm(os.listdir(subdir))):
            logger.info("Working on dataset {} {}".format(subdir,dataset))

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
                else:
                    logger.info("data already initialised")
                observer = Observer.load(os.path.join(subdir, dataset,observer_path))
                logger.info("test with geneo init")
                fig, axes = plt.subplots(2,2)

                cnn_inits = partial(geneo_init,
                                    operators = observer.get_operators())
                cnn_params = {'architecture': 'cnn',
                            'num_units':[64],
                            'kernels': [kernel_shape_layer1],
                            'strides_cnn': [(1,1)],
                            'activations': ['relu'],
                            'strides_pool': [(2,2)],
                            'pools': [(2,2)],
                            'structure': ['conv', 'pool'],
                            'initializers':[cnn_inits],
                            'trainable':[False]}
                fc_params = {'architecture':None,
                             'num_layers': 1,
                             'num_units': [64],
                             'activations': ['relu'],
                             'init_model': None}
                net_params = [cnn_params, fc_params]
                train_params = {
                    'loss': keras.losses.categorical_crossentropy,
                    'optimizer': keras.optimizers.Adam, 'lr': 0.000001,
                    'callbacks': [EarlyStopping(monitor='val_accuracy', min_delta=0.0001,
                                                patience=5, verbose=1, mode='auto')],
                    'batch_size': 32,
                    'epochs': 100,
                    'validation_split': 0.3,
                    'shuffle': True,
                    'metrics': ['accuracy']
                    }
                if len(data.x_train.shape) < 4:
                    data.x_train = np.expand_dims(data.x_train, axis=3)
                input_shape =  data.x_train[0].shape
                logger.info("input_shape {}".format(input_shape))
                net = build_net(arch ="cnn", params =net_params, input_shape = input_shape,
                                num_classes =len(data.classes))
                #net.layers[0].trainable=False
                print(net.layers[0].name)
                print(net.layers[0].trainable)
                net = compile(net, train_parameters =train_params)
                history = net.fit(x=data.x_train,
                                  y=data.y_train,
                                  batch_size=train_params['batch_size'],
                                  epochs=train_params['epochs'],
                                  callbacks=train_params['callbacks'],
                                  validation_split=train_params['validation_split'],
                                  shuffle=train_params['shuffle'],
                                  verbose=0
                                  )

                plot_net_history(history, axs = axes[0])
                logger.info("glorot uniform initialization")
                cnn_params = {'architecture': 'cnn',
                            'num_units':[64],
                            'kernels': [kernel_shape_layer1],
                            'strides_cnn': [(1,1)],
                            'activations': ['relu'],
                            'strides_pool': [(2,2)],
                            'pools': [(2,2)],
                            'structure': ['conv', 'pool'],
                            'initializers':["glorot_uniform"],
                            'trainable':[False]}
                net_params = [cnn_params, fc_params]
                train_params = {
                    'loss': keras.losses.categorical_crossentropy,
                    'optimizer': keras.optimizers.Adam, 'lr': 0.0001,
                    'callbacks': [EarlyStopping(monitor='val_accuracy', min_delta=0.0001,
                                                patience=5, verbose=1, mode='auto')],
                    'batch_size': 32,
                    'epochs': 100,
                    'validation_split': 0.3,
                    'shuffle': True,
                    'metrics': ['accuracy']
                    }
                if len(data.x_train.shape) < 4:
                    data.x_train = np.expand_dims(data.x_train, axis=3)
                input_shape =  data.x_train[0].shape
                logger.info("input_shape {}".format(input_shape))
                net2 = build_net(arch ="cnn", params =net_params, input_shape = input_shape,
                                num_classes =len(data.classes))
                net2 = compile(net2, train_parameters =train_params)
                history2 = net2.fit(x=data.x_train,
                                  y=data.y_train,
                                  batch_size=train_params['batch_size'],
                                  epochs=train_params['epochs'],
                                  callbacks=train_params['callbacks'],
                                  validation_split=train_params['validation_split'],
                                  shuffle=train_params['shuffle'],
                                  verbose=2)
                fig_path = os.path.join(subdir, dataset, observer_path)
                fig_path, _ = os.path.splitext(fig_path)
                fig_path = fig_path + ".svg"
                plot_net_history([history2, history], axs = axes[1],
                                 save_to=fig_path, add_labels = False)
