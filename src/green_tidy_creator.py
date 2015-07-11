from __future__ import division
__author__ = 'Vladimir Iglovikov'

'''
This script reads jpeg images, rescales them, saves green channel and saves to teh disk
'''

from pylab import *

from PIL import Image

X = []

fNames = os.listdir('../data/train')

shape = (128, 128)
#we are worinkg with impages with size 128x128 => new shape as for 1d array will be 16384

count = 0
for fName in fNames:
  img = Image.open(os.path.join("..", "data", "train", fName))

  #rescale image

  red, green, blue = img.split()

  '''
  May be later I will try resizing with antializaing, although not sure that it will be relevant when
  we make size smaller
  '''
  # img = img.resize(size, Image.ANTIALIAS)
  x = green.resize(shape)

  x.save(os.path.join('..', 'data', 'train_green_128', fName))
  if count >= 1000 and count % 1000 == 0:
    print count
  count += 1
