from __future__ import division
__author__ = 'Vladimir Iglovikov'
import cPickle as pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from lasagne.nonlinearities import softmax

print 'reading train'
fName = open('../data/train_green.pkl')
train = pickle.load(fName)
fName.close()

print 'reading labels'
train_ids = list(fName('../data/train_green_ids').read())

scaler = StandardScaler(with_mean=False)

encoder = LabelEncoder()
y = encoder.fit_transform(train_ids.values).astype(np.int32)

print 'scaling X'
X = scaler.fit_transform(train.astype(np.float64)).astype(np.float32)

params = {
  'update_learning_rate': 0.01,
  'update_momentum': 0.9,
  'max_epochs':15
}

num_classes = len(set(y))

print 'num_classes = ', num_classes

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden0', layers.DenseLayer),
        # ('hidden1', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 128, 128),  # 28x28 input pixels per batch
    hidden0_num_units=100,  # number of units in hidden layer
    # hidden1_num_units=50,  # number of units in hidden layer
    output_nonlinearity=softmax,  # output layer uses identity function
    output_num_units=num_classes,  # 1 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=params['update_learning_rate'],
    update_momentum=params['update_momentum'],
    use_label_encoder=True,
    regression=False,  # flag to indicate we're dealing with regression problem
    max_epochs=params['max_epochs'],  # we want to train this many epochs
    verbose=1,
    )

random_state = 42

print 'shuffling'
X, y = shuffle(X, y, random_state=random_state)

print 'fitting'
net1.fit(X, y)