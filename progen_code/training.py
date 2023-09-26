from __future__ import division
from __future__ import print_function
import sys
import tensorflow as tf
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import numpy as np
tf.enable_eager_execution()
import transformer
import argparse
import pdb
import re
from collections import Counter
from tensorflow.python import debug as tf_debug
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
import platform

use_py3 = platform.python_version()[0] == '3'

parser = argparse.ArgumentParser(description='TensorFlow code for generating from CTRL')
parser.add_argument('--model_dir', type=str, default = '/export/share/amadani/ctrl/ckpts/final_dummy_new_alldata_nomask.ckpt', help='location of model checkpoint')
parser.add_argument('--tfrecords_dir', type=str, default = '/export/share/amadani/protein-data/augment_samples/aug_nodrop')
parser.add_argument('--seed', type=int, default=313,
                                        help='random seed for TensorFlow, numpy and PythonHash')
parser.add_argument('--sequence_len', type=int, default=512,
                                        help='sequence len of model being fine-tuned (must match also the TFRecords)')
parser.add_argument('--iterations', type=int, default=1000000,
                                        help='number of iterations (not epochs)')
parser.add_argument('--batch_size', type=int, default = 4)
parser.add_argument('--vocab_loc', type=str, default = 'mapping_files/vocab.txt')
parser.add_argument('--no_loss_4_pad', type=bool, default=True)
parser.add_argument('--weighted_loss', type=bool, default=True)

args = parser.parse_args()

global no_loss_4_pad, weighted_loss
weighted_loss = args.weighted_loss
no_loss_4_pad = args.no_loss_4_pad
if not no_loss_4_pad: print('the PAD tokens have a loss contribution')
if weighted_loss: print('weighting loss by existence tag')

tf.random.set_random_seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)

# load the vocabulary from file
vocab = open(args.vocab_loc).readlines() if not use_py3 else open(args.vocab_loc, encoding='utf-8').read().split('\n')
vocab = list(map(lambda x: x.split(' ')[0], vocab))
global pad_index
pad_index = np.argwhere(np.array(vocab)=='PAD')[0][0]
#print ('{} unique words'.format(len(vocab)))
# length of the vocabulary
vocab_size = len(vocab)
print('---------------------vocab size: ', vocab_size)

# sequence length to use for the transformer
# must match the model being fine-tuned
global seq_length
seq_length = args.sequence_len -1

def input_fn(params=None):
    print('READING!', params)
    dataset = tf.data.Dataset.list_files(tf.io.gfile.glob(os.path.join(args.tfrecords_dir, '*.tfrecords')), shuffle=True)

    tf_data = tf.data.TFRecordDataset(dataset)
    myfeatures = {
        'input': tf.io.FixedLenFeature([seq_length], tf.int64),
        'output': tf.io.FixedLenFeature([seq_length+1], tf.int64) # ATTN: added 1 for existence weight
        }

    def _parse_text_function(example_proto):
        blah = tf.io.parse_single_example(example_proto, myfeatures)
        return blah['input'], blah['output']
    
    train_data = tf_data.map(_parse_text_function).batch(args.batch_size, drop_remainder=True).repeat().shuffle(10000)#.prefetch(tf.contrib.data.AUTOTUNE)
    
    return train_data


# the dimension of the transformer
embedding_dim = 512 # Original - 1280


# Now, we begin defining the model
# we defer the transformer definition to transformer.py
# here, we only define the tied softmax layer
# this layer ties the softmax weights to the input embeddings
class TiedEmbeddingSoftmax(tf.keras.layers.Layer):

  def __init__(self, vocab_size=vocab_size, embedding_size=embedding_dim, **kwargs):
    super(TiedEmbeddingSoftmax, self).__init__()
    self.w = self.add_weight(name='w', shape=(vocab_size, embedding_size),
                             initializer='random_normal',
                             trainable=True)
    self.b = self.add_weight(name='b', shape=(vocab_size,),
                             initializer='zeros',
                             trainable=True)

  def call(self, inputs, embed=True):
    if embed:
      dtype = tf.keras.backend.dtype(inputs)
      if dtype != 'int32' and dtype != 'int64':
        inputs = math_ops.cast(inputs, 'int32')
      return embedding_ops.embedding_lookup(self.w, inputs)
    else:
      return tf.tensordot(inputs, tf.transpose(self.w), 1) + self.b

# input for the keras model
tokens = tf.keras.layers.Input(shape=(seq_length,), dtype='int32')

# instantiates a tied softmax class
tied_embedding_softmax = TiedEmbeddingSoftmax()

# embedded tokens, before passing it to the transformer
embedded = tied_embedding_softmax(tokens, embed=True)

# the activations after passing it from the transformer
# for some odd reason, TPUs don't play well with specifying the arguments of the Encoder() function
# so you have to leave them at their defaults
transformed = transformer.Encoder()(embedded, training=False)


# pass the activations from our tiedsoftmax class
# this time with embed=False denoting that we are doing the softmax operation
# and not a lookup
logits = tied_embedding_softmax(transformed, embed=False)


# finally, define the Keras model with inputs as tokens and outputs as the logits we just computed
model = tf.keras.Model(inputs=tokens, outputs=logits)


# the loss function is a simple categorical crossentropy between the logits and the labels
def loss(labels, logits):
    global no_loss_4_pad, seq_length, weighted_loss, pad_index
    labels, existence = tf.split(labels,[seq_length,1],1)
    labels = tf.cast(tf.squeeze(labels), tf.int32)
    existence = tf.squeeze(existence)
    if not no_loss_4_pad:
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    else:
        raw_prediction=tf.reshape(logits,[-1,vocab_size])
        gt = tf.reshape(labels,[-1])
        indices= tf.cast(tf.not_equal(gt, pad_index), tf.float32)#gt==pad_index #mask, same size as gt
        #gt=tf.cast(tf.gather(gt,indices),tf.int32)
        #prediction=tf.gather(raw_prediction,indices)
        #loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=raw_prediction,labels=gt,name="entropy")
        loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name="entropy")
        loss = tf.reshape(loss,[-1])
        loss=loss*indices
        loss = tf.reshape(loss,[-1,seq_length])
        #loss=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,labels=gt,name="entropy")
        loss=tf.reduce_mean(tf.multiply(tf.reduce_mean(loss, 1), tf.cast(existence, tf.float32)))
    return loss

# the optimizer is not used since this code only supports inference
# however, to compile the model, we still define it
optimizer = tf.contrib.estimator.clip_gradients_by_norm(
        tf.train.AdagradOptimizer(learning_rate=1e-2), 0.25)


# compile the model with the optimizer and loss
#logging_hook = tf.train.LoggingTensorHook({"loss":loss}, every_n_iter = 10)
model.compile(optimizer=optimizer, loss=loss)
print(model.summary())


# IMPORTANT
# this is where the saved model is presented to the code
# the model directory should have the model checkpoint and
# a checkpoint file
run_config = tf.contrib.tpu.RunConfig(
        model_dir=args.model_dir)


# this converts the Keras model to a TensorFlow estimator
# this step is critical
# remember to patch the TF 1.14 file before running the code, else you're going to see errors here

run_config = tf.contrib.tpu.RunConfig(
        model_dir=args.model_dir,
        session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
        save_summary_steps = 2, log_step_count_steps = 10,
        tpu_config=tf.contrib.tpu.TPUConfig(iterations_per_loop=100, num_cores_per_replica=1, input_partition_dims=[[1, 1], [1, 1]], per_host_input_for_training=3))
tf.logging.set_verbosity(tf.logging.INFO) # INFO , tf.logging.ERROR

estimator_model = tf.keras.estimator.model_to_estimator(keras_model=model, config=run_config)

estimator_model.train(input_fn=input_fn, steps=args.iterations)#, hooks = [logging_hook])
