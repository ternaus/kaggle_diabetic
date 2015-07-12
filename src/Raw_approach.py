from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
I do not have that much backgroun with this tricky image processing.
Naive aproach with just throwing everything in, did not work so well.

Naive approach gave me 0.88 on teh holdout set and 0 ad LB

Let's see if I can get better
'''

import cPickle as pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from sklearn.preprocessing import LabelEncoder
from lasagne.nonlinearities import softmax
import matplotlib.image as mpimg
from kappa_squared import quadratic_weighted_kappa
import theano
from sklearn import cross_validation
from sklearn import metrics

import os
import pandas as pd
# print 'reading train'
# fName = open('../data/train_green.pkl')
# train = pickle.load(fName)
# fName.close()
from pylab import *
import seaborn as sns

from PIL import Image
from sklearn.utils import shuffle

def float32(k):
    return np.cast['float32'](k)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

class AdaptiveVariable(object):
    def __init__(self, name, start=0.03, stop=0.000001, inc=1.1, dec=0.5):
        self.name = name
        self.start, self.stop = start, stop
        self.inc, self.dec = inc, dec

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        if len(train_history) > 1:
            previous_valid = train_history[-2]['valid_loss']
        else:
            previous_valid = np.inf
        current_value = getattr(nn, self.name).get_value()
        if current_value < self.stop:
            raise StopIteration()
        if current_valid > previous_valid:
            getattr(nn, self.name).set_value(float32(current_value*self.dec))
        else:
            getattr(nn, self.name).set_value(float32(current_value*self.inc))

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


X = []

fNames = os.listdir('../data/train_green_128')

y = pd.DataFrame()
y['image'] = fNames
y['image'] = y['image'].apply(lambda x: x.rstrip('.jpeg'))

y = (y
     .merge(pd.read_csv('../data/trainLabels.csv'))['level']
     .values
     )


shape = (128, 128)
#we are worinkg with impages with size 128x128 => new shape as for 1d array will be 16384

count = 0
for fName in fNames:
  img = Image.open(os.path.join("..", "data", "train_green_128", fName))

  X.append(np.reshape(np.asarray(img), 16384))

  if count >= 1000 and count % 1000 == 0:
    print count
  count += 1


X = np.array(X)
print X.shape

print 'reading labels'
# train_ids = list(fName('../data/train_green_ids').read())
train_ids = y

scaler = StandardScaler(with_mean=False)

encoder = LabelEncoder()
y = encoder.fit_transform(train_ids).astype(np.int32)

print 'scaling X'
X = scaler.fit_transform(np.array(X).astype(np.float64)).astype(np.float32)


X = X.reshape(X.shape[0], 128, 128)

print X.shape
print len(y)
params = {
  'update_learning_rate': 0.01,
  'update_momentum': 0.9,
  'max_epochs': 100
}

num_classes = len(set(y))

print 'num_classes = ', num_classes

clf = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden0', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),
        ('hidden1', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 128, 128),  # 128x128 input pixels per batch
    hidden0_num_units=500,  # number of units in hidden layer
    dropout1_p=0.5,
    hidden1_num_units=500,  # number of units in hidden layer
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
    update_learning_rate=theano.shared(float32(0.03)),
                 on_epoch_finished=[
                    AdaptiveVariable('update_learning_rate', start=0.001, stop=0.00001),
                    AdjustVariable('update_momentum', start=0.9, stop=0.999),
                    EarlyStopping(),
                ]
    )

random_state = 42
kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better=True)

print 'estimating'
skf = cross_validation.StratifiedKFold(y, n_folds=5, shuffle=True, random_state=random_state)
# scores = cross_validation.cross_val_score(clf, X, y, cv=skf)
scores = cross_validation.cross_val_score(clf, X, y, cv=skf, scoring=kappa_scorer, n_jobs=-1)
print np.mean(scores), np.std(scores)

#
# print 'shuffling'
# X, y = shuffle(X, y, random_state=random_state)
#
# print 'fitting'
# net1.fit(X, y)
#
# train_loss = np.array([i["train_loss"] for i in net1.train_history_])
# valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])
#
# plot(train_loss, linewidth=3, label='train')
# plot(valid_loss, linewidth=3, label='valid')
# yscale("log")
#
# legend()
#
# savefig('plots/double.png')

# ind = True
ind = False

if ind:
  X_test = []

  fNames = os.listdir('../data/test')
  count = 0

  for fName in fNames:
    img = mpimg.imread(os.path.join("..", "data", "test", fName))[:, :, 0]
    img = np.reshape(img, 16384)

    X_test.append(img.tolist())
    # X.append(img)
    if count >= 1000 and count % 1000 == 0:
      print count
    count += 1

  print 'scaling test'

  X_test = scaler.transform(np.array(X_test).astype(np.float64)).astype(np.float32)
  print 'reshaping test'
  X_test = X_test.reshape(X_test.shape[0], 128, 128)

  print 'creating submission'
  submission = pd.DataFrame()
  submission['image'] = submission['image'].apply(lambda x: x.rstrip('.jpeg'))
  submission['level'] = net1.predict(X_test)
  submission.to_csv('predictions/double.csv', index=False)
