from __future__ import division
__author__ = 'Vladimir Iglovikov'

import matplotlib.image as mpimg
import os
import numpy as np
import pandas as pd
from pylab import *
import cPickle as pickle
'''
This scrip creates dataframe from images
'''

result = []

fNames = os.listdir('../data/train')

#we are worinkg with impages with size 128x128 => new shape as for 1d array will be 16384
#I am keeping only layer 0

count = 0
for fName in fNames:
  img = mpimg.imread(os.path.join("..", "data", "train", fName))[:, :, 0]
  # img = np.reshape(img, 16384)
  result.append(img.tolist())
  if count >= 100 and count % 100 == 0:
    print count
  count += 1

fName = open('../data/train_green.pkl', 'w')
# print >> fName, result

pickle.dump(result, fName)
fName.close()

fName = open('../data/train_green_ids', 'w')
print >> fName, fNames
fName.close()


#
# result = pd.DataFrame(result)
# print result.shape
# result['label'] = fNames
# result['label'] = result['label'].apply(lambda x: x.rstrip('.jpeg'), 1)
# result.to_csv('../data/train_green.csv', index=False)