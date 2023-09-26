import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random
import tensorflow as tf
import tqdm
import pdb
import glob
import sys
import re
import argparse
import platform

from transformProtein import transformProtein

savefold = '/export/share/amadani/protein-data/augment_samples/aug_0/'
obj = transformProtein(selectSwiss = 0.9, selectTrembl = 0.9,
                       maxTaxaPerSample = 3, maxKwPerSample = 5,
                       dropRate = 0.2)

t1 = time.time()
for chunknum in range(141):
    with open('/export/share/amadani/protein-data/train_test/train'+str(chunknum)+'.p','rb') as handle:
        train_chunk = pickle.load(handle)
    existences = []
    with tf.io.TFRecordWriter(savefold+'augtrain'+str(chunknum)+'.tfrecords') as writer:
        for uid in train_chunk.keys():
            sample_arr, existence = obj.transformSample(train_chunk[uid])
            inputs = sample_arr[:-1]
            outputs = sample_arr[1:]
            
            # for weighting loss based on existence
            if existence in set({0,1}):
                existence = 2
            else:
                existence = 1
            outputs.append(existence) # add existence tag at the end of outputs
            example_proto = tf.train.Example(features=tf.train.Features(feature={'input': tf.train.Feature(int64_list=tf.train.Int64List(value=inputs)),
                                                                                 'output': tf.train.Feature(int64_list=tf.train.Int64List(value=outputs))}))
            writer.write(example_proto.SerializeToString())
    print(chunknum, 'time',time.time()-t1)
    t1 = time.time()
