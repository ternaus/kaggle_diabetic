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

  X.append(np.reshape(np.asarray(img)), 16384)

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

net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden0', layers.DenseLayer),
        ('hidden1', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 128, 128),  # 128x128 input pixels per batch
    hidden0_num_units=500,  # number of units in hidden layer
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
    )

random_state = 42

print 'shuffling'
X, y = shuffle(X, y, random_state=random_state)

print 'fitting'
net1.fit(X, y)

train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])

plot(train_loss, linewidth=3, label='train')
plot(valid_loss, linewidth=3, label='valid')
yscale("log")

legend()

savefig('plots/double.png')

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
