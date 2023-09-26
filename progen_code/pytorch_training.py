from __future__ import print_function
from __future__ import division
import sys
import torch
import os
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

use_py3 = platform.python_version()[0] == '3'

parser = argparse.ArgumentParser(description='TensorFlow code for generating from CTRL')
parser.add_argument('--model_dir', type =str, default='model_v0.pth',
                                        help='location of training model checkpoint')
parser.add_argument('--model_path', type=str, default='/home/amadani/ctrl/ckpt/seqlen256_36layers_v0.ckpt/model.ckpt-684000', help='location of model *data* checkpoint to load; this is NOT the directory but rather the model checkpoint')
parser.add_argument('--seed', type=int, default=313,
                                        help='random seed for TensorFlow, numpy and PythonHash')
parser.add_argument('--sequence_len', type=int, default=511,
                                        help='sequence len of model being fine-tuned')
parser.add_argument('--num_epochs', type=int, default=10000, help='number of epochs to train for')
parser.add_argument('--num_layers', type=int, default=36, help='number of transfomer layers. used for loading checkpoint')
parser.add_argument('--batch_size', type=int, default = 4, help='batch size for dataloader')
parser.add_argument('--vocab_loc', type=str, default='mapping_files/vocab.txt', help='vocab location')
parser.add_argument('--num_workers', type=int, default=0, help='for dataloader')
parser.add_argument('--warmup_iteration', type=int, default=1000, help='LR warmup cutoff')
parser.add_argument('--save_iter', type=int, default=1000, help='save model checkpoint every X iterations')

args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
np.random.seed(args.seed)

# load the vocabulary from file
vocab = open(args.vocab_loc).readlines() if not use_py3 else open(args.vocab_loc, encoding='utf-8').read().split('\n')[:-1]
vocab = list(map(lambda x: x.split(' ')[0], vocab))
# length of the vocabulary
vocab_size = len(vocab)
print('-----vocab size',vocab_size,'------')

# define the numericalization map
# idx2word maps the numericalized ID to the word
# word2idx maps the word to the numericalized ID
#word2idx = {u:i for i, u in enumerate(vocab)}
#idx2word = np.array(vocab)

# sequence length to use for transfomer
seq_length = args.sequence_len

embedding_dim = 1280

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
      #self.tied_embedding_softmax.w = torch.nn.Parameter(torch.tensor(reader.get_tensor('w')).to('cuda'))
     #self.tied_embedding_softmax.b = torch.nn.Parameter(torch.tensor(reader.get_tensor('b')).to('cuda'))

      list_of_variables = list(filter(lambda x: 'Adagrad' not in x, reader.get_variable_to_shape_map().keys()))

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
      torch.save({
        'softmax': self.tied_embedding_softmax.state_dict(),
        'encoder': self.encoder.state_dict(),
      }, pytorch_model_hash)

# initialize ctrl object
# load checkpoint with args.model_path
model = CTRLmodel()
print('model initialized')
model.loadCheckpoint(model_path=args.model_path, num_layers = args.num_layers)
print('previous checkpoint loaded')
model = model.cuda()

# freeze all weights except embedding
for p in model.parameters():
    p.requires_grad=False
model.tied_embedding_softmax.w.requires_grad=True
model.tied_embedding_softmax.b.requires_grad=True

class Trainer(object):
    def __init__(self, model, warmup_iteration, seq_length, batch_size, num_workers, vocab_size, model_dir, save_iter):
        self.model = model
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_size = vocab_size
        self.model_dir = model_dir
        self.save_iter = save_iter
        self.firstAAidx = self.vocab_size - 26 # Assuming that the pad token is the last token and AAs are at the end
        
        self.optimizer = torch.optim.Adam(self.model.parameters()) #lr, betas
        lambdafn = lambda iteration: min(iteration/(warmup_iteration*1.0),1.0)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambdafn)
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.vocab_size-1, reduction='none')
        
        self.transformFull = transformProtein(maxSampleLength = seq_length+1, 
                                              selectSwiss = 1.0, selectTrembl = 1.0, 
                                              maxTaxaPerSample = 3, maxKwPerSample = 5, dropRate = 0.0)
        self.transformPartial = transformProtein(maxSampleLength = seq_length+1,   
                                              selectSwiss = 0.9, selectTrembl = 0.9,
                                              maxTaxaPerSample = 3, maxKwPerSample = 5, dropRate = 0.2)
        self.transformNone = transformProtein(maxSampleLength = seq_length+1,   
                                              selectSwiss = 1.0, selectTrembl = 1.0,
                                              maxTaxaPerSample = 0, maxKwPerSample = 0, dropRate = 1.0)
        
        self.writer = SummaryWriter()

    def train(self, num_epochs):
        self.model.train()

        iter_num = 0
        for epoch in range(num_epochs):
            loss_e = 0.0
            num_e = 0

            for chunknum in range(141):
                pklpath = '/home/amadani/proteinlm/data/train_test_pkl/'
                pklpath = pklpath + 'train' + str(chunknum) + '.p'
                chunk_dataset = ProteinDataset(pklpath, firstAAidx = self.firstAAidx, transformFull = self.transformFull, 
                                               transformPartial = self.transformPartial, transformNone = self.transformNone)
                dataloader = DataLoader(chunk_dataset, shuffle = True, batch_size = self.batch_size,
                                        num_workers = self.num_workers, pin_memory = False) #TODO pinmem?
                
                for i, (sample, labels, existence, padIndex, begAAindex) in enumerate(dataloader):
                    self.optimizer.zero_grad()
                    sample, labels, existence, padIndex = sample.cuda(), labels.cuda(), existence.cuda(), padIndex.cuda()
                    output = self.model(sample)
                    #pdb.set_trace()
                    loss = self.criterion(output.permute(0,2,1), labels)
                    loss = torch.mean((torch.sum(loss,dim=1)/padIndex)*existence) #pad masking, loss weighting
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
                    self.optimizer.step()
                    self.scheduler.step()
                    loss_e += loss.item()
                    num_e += sample.shape[0]
                    iter_num += 1
                    self.writer.add_scalar('Loss_iteration',loss.item(),iter_num)

                    if (i+1)%self.save_iter==0:
                        torch.save({'epoch': epoch, 'chunknum': chunknum, 'iteration':i,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                    'loss': loss,
                                   }, self.model_dir)
                loss_e/=num_e
            #print(loss_e)
            #self.writer.add_scalar('Loss_epoch',loss_e, epoch)

training = Trainer(model=model, warmup_iteration=args.warmup_iteration, seq_length=seq_length,
                   batch_size=args.batch_size, num_workers=args.num_workers, vocab_size=vocab_size,
                   model_dir = args.model_dir, save_iter=args.save_iter)
print('begin training...')
training.train(args.num_epochs)



