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
import matplotlib.image as mpimg
import os
import pandas as pd
# print 'reading train'
# fName = open('../data/train_green.pkl')
# train = pickle.load(fName)
# fName.close()
from pylab import *
import seaborn as sns


X = []

fNames = os.listdir('../data/train')

y = pd.DataFrame()
y['image'] = fNames
y['image'] = y['image'].apply(lambda x: x.rstrip('.jpeg'))

y = (y
     .merge(pd.read_csv('../data/trainLabels.csv'))['level']
     # .values
     )


#we are worinkg with impages with size 128x128 => new shape as for 1d array will be 16384
#I am keeping only layer 0

count = 0
for fName in fNames:
  img = mpimg.imread(os.path.join("..", "data", "train", fName))[:, :, 0]
  img = np.reshape(img, 16384)

  X.append(img.tolist())
  # X.append(img)
  if count >= 1000 and count % 1000 == 0:
    print count
  count += 1



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
  'max_epochs':100
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
    input_shape=(None, 128, 128),  # 128x128 input pixels per batch
    hidden0_num_units=100,  # number of units in hidden layer
    # hidden1_num_units=100,  # number of units in hidden layer
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
savefig('plots/simple.png')
legend()