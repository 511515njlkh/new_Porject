from __future__ import print_function
from __future__ import division
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
import torch
import tqdm
import pdb
import numpy as np
import platform
import hashlib
import pytorch_transformer
import re
import argparse
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from transformProtein import transformProtein
from ProteinDataset import ProteinDataset
from torch.utils.data import Dataset, DataLoader
import pickle
import time

load_model_path = '/home/amadani/proteinlm/model_nov19/' # just the folder itself

seq_length = 511
embedding_dim = 1280
num_layers = 36
vocab_loc = '/home/amadani/proteinlm/protein-lm/mapping_files/vocab.txt'

num_workers = 0
batch_size = 20
num_batches_eval = 300

use_py3 = platform.python_version()[0] == '3'
vocab = open(vocab_loc).readlines() if not use_py3 else open(vocab_loc, encoding='utf-8').read().split('\n')[:-1]
vocab = list(map(lambda x: x.split(' ')[0], vocab))
vocab_size = len(vocab)
print('-----vocab size',vocab_size,'------')

class TiedEmbeddingSoftmax(torch.nn.Module):

  def __init__(self, vocab_size=vocab_size, embedding_size=embedding_dim, **kwargs):
    super(TiedEmbeddingSoftmax, self).__init__()
    self.w = torch.nn.Parameter(torch.normal(0., 1e-2, size=(vocab_size, embedding_size)))
    self.b = torch.nn.Parameter(torch.zeros(vocab_size))

  def forward(self, inputs, embed=True):
    if embed:
      return torch.nn.functional.embedding(inputs, self.w)
    else:
      return torch.tensordot(inputs, self.w.t(), 1) + self.b

class CTRLmodel(torch.nn.Module):
  def __init__(self):
    super(CTRLmodel,self).__init__()
    self.tied_embedding_softmax = TiedEmbeddingSoftmax()
    self.encoder = pytorch_transformer.Encoder()

  def forward(self, inputs):
    x = self.tied_embedding_softmax(inputs, embed = True)
    x = self.encoder(x)
    x = self.tied_embedding_softmax(x, embed = False)
    return x

  def loadCheckpoint(self, model_path, num_layers):
    pytorch_model_hash = hashlib.md5(model_path.encode('utf-8')).hexdigest()

    if os.path.exists(pytorch_model_hash):
      print('Found PyTorch checkpoint @', pytorch_model_hash)
      print('Loading instead of converting from TensorFlow')
      checkpoint = torch.load(pytorch_model_hash)
      self.tied_embedding_softmax.load_state_dict(checkpoint['softmax'])
      self.encoder.load_state_dict(checkpoint['encoder'])

      self.tied_embedding_softmax.to('cuda')
      self.encoder.to('cuda')

    else:
      print('Could not find PyTorch checkpoint')
      print('Converting weights and will store the PyTorch checkpoint as ', pytorch_model_hash)
      chkpt_for_reader = model_path # '.'.join(model_path.split('.')[:-1])
      reader = pywrap_tensorflow.NewCheckpointReader(chkpt_for_reader)

      self.tied_embedding_softmax.w = torch.nn.Parameter(torch.tensor(reader.get_tensor('w')).to('cuda'))
      self.tied_embedding_softmax.b = torch.nn.Parameter(torch.tensor(reader.get_tensor('b')).to('cuda'))

      list_of_variables = list(filter(lambda x: ('Adagrad' not in x) and ('Adam' not in x), reader.get_variable_to_shape_map().keys()))

      str2parameter = lambda x: torch.nn.Parameter(torch.tensor(reader.get_tensor(x)).t().to('cuda'))
      
      self.encoder.layernorm.weight = str2parameter('encoder/layer_normalization_'+str(int(num_layers*2))+'/gamma')
      self.encoder.layernorm.bias = str2parameter('encoder/layer_normalization_'+str(int(num_layers*2))+'/beta')
      for i in tqdm.tqdm(range(num_layers)):
        if i==0:
          layer_variables = sorted(filter(lambda x: 'layer/' in x, list_of_variables))
        else:
          layer_variables = sorted(filter(lambda x: 'layer_'+str(i)+'/' in x, list_of_variables))
        
        current_layer = getattr(self.encoder, 'layer'+str(i))
        
        current_layer.layernorm1.bias = str2parameter(layer_variables[0])
        current_layer.layernorm1.weight = str2parameter(layer_variables[1])

        current_layer.layernorm2.bias = str2parameter(layer_variables[2])
        current_layer.layernorm2.weight = str2parameter(layer_variables[3])


        current_layer.multi_head_attention.Wq.bias = str2parameter(layer_variables[4])
        current_layer.multi_head_attention.Wq.weight = str2parameter(layer_variables[5])
        current_layer.multi_head_attention.Wk.bias = str2parameter(layer_variables[6])
        current_layer.multi_head_attention.Wk.weight = str2parameter(layer_variables[7])
        current_layer.multi_head_attention.Wv.bias = str2parameter(layer_variables[8])
        current_layer.multi_head_attention.Wv.weight = str2parameter(layer_variables[9])
        current_layer.multi_head_attention.dense.bias = str2parameter(layer_variables[10])
        current_layer.multi_head_attention.dense.weight = str2parameter(layer_variables[11])
        current_layer.ffn[0].bias = str2parameter(layer_variables[12])
        current_layer.ffn[0].weight = str2parameter(layer_variables[13])
        current_layer.ffn[2].bias = str2parameter(layer_variables[14])
        current_layer.ffn[2].weight = str2parameter(layer_variables[15])
      
      #torch.save({'softmax': self.tied_embedding_softmax.state_dict(),'encoder': self.encoder.state_dict(),}, 
      #           pytorch_model_hash)




transformObj = transformProtein(maxSampleLength = seq_length+1, selectSwiss = 1.0, selectTrembl = 1.0, 
                                maxTaxaPerSample = 3, maxKwPerSample = 5, dropRate = 0.0)

def getBLOSUM(matrix_filename = '/home/amadani/proteinlm/protein-lm/blosum62.txt'):
    with open(matrix_filename) as matrix_file:
        matrix = matrix_file.read()
    lines = matrix.strip().split('\n')

    header = lines.pop(0)
    columns = header.split()
    matrix = {}

    for row in lines:
        entries = row.split()
        row_name = entries.pop(0)
        matrix[row_name] = {}

        if len(entries) != len(columns):
            raise Exception('Improper entry number in row')
        for column_name in columns:
            matrix[row_name][column_name] = entries.pop(0)
    return matrix
blosum = getBLOSUM()

# add U and O amino acids. U substitutes with C. O substitutes with K. make it symmetric.
# remove '*' entries
# change blosum['X']['X'] with 1
del blosum['*']
blosum['X']['X'] = 1
for i in blosum.keys():
    del blosum[i]['*']
    for j in blosum[i].keys():
        blosum[i][j]=int(blosum[i][j])
    blosum[i]['O']= -3
    blosum[i]['U']= -3
blosum['O']={i:-3 for i in blosum.keys()}
blosum['U']={i:-3 for i in blosum.keys()}

blosum['O']['O']=2
blosum['O']['K']=1
blosum['K']['O']=0

blosum['U']['U']=2
blosum['U']['C']=1
blosum['C']['U']=0

ckpt_nums = set([re.split('-|\.',i)[2] for i in os.listdir(load_model_path) if i.startswith('model.ckpt')])

if os.path.exists(load_model_path+'evals.p'):
    with open(load_model_path+'evals.p','rb') as handle:
        evals_dict = pickle.load(handle)
else:
    evals_dict = {}

for trainortest in ['train0.p','test.p']:
    for ckptnum in ckpt_nums:
        curr_model_path = load_model_path+'model.ckpt-'+ckptnum
        # check if ckpt has been evaluated already
        if (ckptnum in evals_dict) and (trainortest in evals_dict[ckptnum]):
            continue

        model = CTRLmodel()
        print('model initialized')
        reader = model.loadCheckpoint(model_path=curr_model_path, num_layers = num_layers)
        print('previous checkpoint loaded')
        model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters()) #lr, betas


        pklpath = '/home/amadani/proteinlm/data/train_test_pkl/'
        pklpath = pklpath + trainortest
        firstAAidx = vocab_size - 26

        chunk_dataset = ProteinDataset(pklpath, firstAAidx = firstAAidx, evalTransform = transformObj)
        dataloader = DataLoader(chunk_dataset, shuffle = False, batch_size = batch_size,
                                num_workers = num_workers, pin_memory = True) #TODO pinmem?

        model.eval()
        ppls = []
        hard_accs = []
        soft_accs = []

        soft_accs_along_seq = {i:[] for i in range(300)}

        t1 = time.time()
        with torch.no_grad():
            for i, (sample, labels, existence, padIndex, begAAindex) in zip(range(num_batches_eval),dataloader):
                sample = sample.cuda()
                output = model(sample)

                output = output[:,:,:-1] # remove pad token logits

                for sample_i in range(output.shape[0]): # each sample in batch
                    y_pred, y_actual = output[sample_i], labels[sample_i]
                    y_pred = y_pred[begAAindex[sample_i]:padIndex[sample_i]-1]
                    y_actual = y_actual[begAAindex[sample_i]:padIndex[sample_i]-1]

                    y_pred = F.softmax(y_pred,dim=1)

                    # calculate perplexity
                    ent = torch.log(y_pred[0][y_actual[0]])
                    for ii in range(1,y_pred.shape[0]): # each token in available tokens
                        ent+=torch.log(y_pred[ii][y_actual[ii]])
                    ppl = torch.exp(-ent/y_pred.shape[0])
                    ppls.append(ppl.item())            

                    # below needs to convert to cpu so will slow down calculation
                    y_pred = y_pred.cpu().numpy()
                    y_actual = y_actual.numpy()

                    # examine distribution of predicted AAs
                    if (i==0) and (sample_i==0):
                        aa_probs = np.sum(y_pred,axis=0)[-25:]
                        total_num_AAs = y_pred.shape[0]
                    else:
                        aa_probs += np.sum(y_pred,axis=0)[-25:]
                        total_num_AAs += y_pred.shape[0]

                    # calculate hard accuracy
                    y_pred_idx = np.argmax(y_pred,axis=1)
                    hard_accs.append(sum(y_pred_idx==y_actual)*1.0/y_pred_idx.shape[0])

                    # calculate soft accuracy
                    tp = 0.
                    for iii in range(y_actual.shape[0]):
                        if y_pred_idx[iii] >= len(vocab)-26:
                            if blosum[vocab[y_actual[iii]]][vocab[y_pred_idx[iii]]] >= 0:
                                tp += 1
                                # for accuracy along seq
                                if iii in soft_accs_along_seq:
                                    soft_accs_along_seq[iii].append(1)
                            elif iii in soft_accs_along_seq:
                                soft_accs_along_seq[iii].append(0)
                    soft_accs.append(tp/y_actual.shape[0])


                print(ckptnum,trainortest.split('.')[0], 'batch ',i,'\tTime',str(round(time.time()-t1,3)), '\tMean ppl',str(round(np.mean(ppls),3)),'\tMean hard acc',
                      str(round(np.mean(hard_accs),3)),'\tMean soft acc',str(round(np.mean(soft_accs),3)))
                t1=time.time()
        # save dict
        sample_dict = {'ppls':ppls, 'hard_accs':hard_accs, 'soft_accs':soft_accs, 
                       'soft_accs_along_seq':soft_accs_along_seq, 'aa_probs':aa_probs}
                       
        if ckptnum in evals_dict:
            evals_dict[ckptnum][trainortest]=sample_dict
        else:
            evals_dict[ckptnum]={trainortest:sample_dict}
        with open(load_model_path+'evals.p','wb') as handle:
            pickle.dump(evals_dict,handle)
