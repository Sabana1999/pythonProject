from future import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

print(tf.version)
import json
import numpy as np
import os
from random import shuffle

datadir = './export'
fulldir = os.path.join(os.getcwd(), datadir)
allfiles = os.listdir(fulldir)

dataset = []

for filename in allfiles:
    filepath = os.path.join(datadir, filename)
    name = './export/' + filename.split(".")[0]
    person = json.dumps(name)
    dataset.append(person)
    #print(dataset)

X =[]
y = []

for person in dataset:
    person_dict = json.loads(person)
    print(person_dict)
    if not isinstance(person_dict, dict):
       continue
    spectogram = person_dict.get('spectogram')
    spectogram = spectogram / np.float32(255)  # normalize input pixels
    print(spectogram)
    if spectogram is None:
        continue
    status = int(person_dict.get('status'))
    print(status)
    X.append(spectogram)
    y.append(status)
print(X)
X = np.array(X)

y = np.array(y)
X = X.reshape((len(X), 28, 28, 1))
print('X shape: ', X.shape, 'y shape:',y.shape)