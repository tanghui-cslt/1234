import keras
from keras.callbacks import EarlyStopping

"""List of standard datasets managed by Keras
"""
AVAILABLE_DATASETS = ['mnist', 'fashion_mnist', 'cifar10', 'cifar100']

"""Default preprocessing parameters
"""
PREPROC_PARAMS = {'bw':False, 'threshold':.66,
                  'reshape':False, 'image_target':(256,256),
                  'blur':False, 'kernel_size':(5,5),
                  'standardize':True}

"""Isometries data augmentation parameter dicts
"""
ROTATION_PARAMS = {'range_theta':[15, 45]}
TRANSLATION_PARAMS = {'rangex':[0, 2], 'rangey':[0, 2]}

"""Persistence params
"""
PERSISTENCE_PARAMS = {'reverse' :False, 'field' :2, 'method':'clearing',
                      'progress':False}

"""Artificial neural networks
"""
cnn_init = keras.initializers.glorot_normal()
CNN_DICT = {'architecture':'cnn','num_units':[16, 64, 100],
            'kernels':[(5,5),(5,5),(5,5)], 'strides_cnn':[(1,1),(1,1),(1,1)],
            'activations':['relu', 'relu', 'relu'],
            'strides_pool':[(2,2),(2,2),(2,2)], 'pools':[(2,2),(2,2),(2,2)],
            'structure':['conv', 'pool','conv', 'pool','conv', 'pool'],
            'initializers':[cnn_init, cnn_init, cnn_init],
            'trainable':[True, True, True]}

FC_DICT = {'architecture':'fc', 'num_layers':2, 'num_units':[64, 100],
           'activations':['relu', 'relu'], 'init_model':None}

TRAIN_PARAMS = {'loss':keras.losses.categorical_crossentropy,
                'optimizer':keras.optimizers.Adam, 'lr':0.001,
                'callbacks':[EarlyStopping(monitor='val_acc', min_delta=0.0001,
                              patience=5, verbose=1, mode='auto')],
                'batch_size':32, 'epochs':100, 'validation_split':0.33,
                'shuffle':True, 'metrics':['accuracy']}
