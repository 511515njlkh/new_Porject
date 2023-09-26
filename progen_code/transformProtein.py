import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import random

class transformProtein:
    def __init__(self, mapfold = 'mapping_files/', maxSampleLength = 512, selectSwiss = 1.0,
                 selectTrembl = 1.0, verbose = False, maxTaxaPerSample = 3, 
                 maxKwPerSample = 5, dropRate = 0.0, seqonly = False, noflipseq = False):

        self.maxSampleLength = maxSampleLength
        self.selectSwiss = selectSwiss
        self.selectTrembl = selectTrembl
        self.verbose = verbose
        self.maxTaxaPerSample = maxTaxaPerSample
        self.maxKwPerSample = maxKwPerSample
        self.dropRate = dropRate
        self.seqonly = seqonly
        self.noflipseq = noflipseq

        if self.seqonly:
            with open(os.path.join(mapfold,'aa_to_ctrl_idx_seqonly.p'),'rb') as handle:
                self.aa_to_ctrl = pickle.load(handle)
            print('Sequence only - no CTRL codes')
            self.oneEncoderLength = max(self.aa_to_ctrl.values())+1
        else:
            with open(os.path.join(mapfold,'kw_to_ctrl_idx.p'),'rb') as handle:
                self.kw_to_ctrl = pickle.load(handle)
            with open(os.path.join(mapfold,'taxa_to_ctrl_idx.p'),'rb') as handle:
                self.taxa_to_ctrl = pickle.load(handle)
            with open(os.path.join(mapfold,'aa_to_ctrl_idx.p'),'rb') as handle:
                self.aa_to_ctrl = pickle.load(handle)
            with open(os.path.join(mapfold,'taxa_to_parents.p'),'rb') as handle: # TODO: remove
                self.taxa_to_parents = pickle.load(handle)
            with open(os.path.join(mapfold,'kw_to_parents.p'),'rb') as handle: # TODO: remove
                self.kw_to_parents = pickle.load(handle)
            with open(os.path.join(mapfold,'kw_to_lineage.p'),'rb') as handle:
                self.kw_to_lineage = pickle.load(handle)
            with open(os.path.join(mapfold,'taxa_to_lineage.p'),'rb') as handle:
                self.taxa_to_lineage = pickle.load(handle)
    
            self.oneEncoderLength = max(max(self.kw_to_ctrl.values()),max(self.taxa_to_ctrl.values()),max(self.aa_to_ctrl.values())) + 1
            print('Using one unified encoder to represent protein sample with length', self.oneEncoderLength)
    
    def transformSeq(self, seq):
        """
        Transform the amino acid sequence. Currently only reverses seq--eventually include substitutions/dropout
        """
        if self.noflipseq:
            return seq
        if np.random.random()>=0.5:
            seq = seq[::-1]
        return seq

    def transformKwSet(self, kws, drop = 0.2):
        """
        Filter kws, dropout, and replace with lineage (including term at end)
        """
        kws = [i for i in kws if i in self.kw_to_ctrl]
        np.random.shuffle(kws)
        kws = kws[:self.maxKwPerSample]
        
        kws = [random.choice(self.kw_to_lineage[i]) for i in kws if np.random.random()>drop]
      
        return kws

    def transformTaxaSet(self, taxa, drop = 0.1):
        """
        Filter taxa, dropout, and replace with lineage (including term at end)
        """
        taxa = [i for i in taxa if i in self.taxa_to_ctrl]
        np.random.shuffle(taxa)
        taxa = taxa[:self.maxTaxaPerSample]
        
        taxa = [self.taxa_to_lineage[i] for i in taxa if np.random.random()>drop]
        
        return taxa

    def transformSample(self, proteinDict, justidx = True):
        """
        Function to transform/augment a sample.
        If it's not in swiss or trembl, existence set to 3. Or if it's found in swiss/trembl and you sample an other taxa.
        Padding with all zeros
        Returns an encoded sample (taxa's,kw's,sequence) and the existence level
        """
        existence = 3
        if (not self.seqonly) and (proteinDict['swiss']!={}) and (np.random.random()<self.selectSwiss):
            ttype = 'swiss'
            uniprotKBid = random.choice(tuple(proteinDict[ttype].keys()))
            existence = proteinDict[ttype][uniprotKBid]['ex']
            kws = self.transformKwSet(proteinDict[ttype][uniprotKBid]['kw'], drop = self.dropRate)
            taxa = self.transformTaxaSet(proteinDict[ttype][uniprotKBid]['taxa'], drop = self.dropRate)
        elif (not self.seqonly) and (proteinDict['trembl']!={}) and (np.random.random()<self.selectTrembl):
            ttype = 'trembl'
            uniprotKBid = random.choice(tuple(proteinDict[ttype].keys()))
            existence = proteinDict[ttype][uniprotKBid]['ex']
            kws = self.transformKwSet(proteinDict[ttype][uniprotKBid]['kw'], drop = self.dropRate)
            taxa = self.transformTaxaSet(proteinDict[ttype][uniprotKBid]['taxa'], drop = self.dropRate)
        elif (not self.seqonly):
            kws = {}
            taxa = self.transformTaxaSet(proteinDict['other_taxa'], drop = self.dropRate)
        seq = self.transformSeq(proteinDict['seq'])

        if self.seqonly:
            encodedSample = []
            seq_idx = 0
            while (len(encodedSample)<self.maxSampleLength) and (seq_idx<len(seq)):
                encodedSample.append(self.aa_to_ctrl[seq[seq_idx]])
                seq_idx += 1
            while len(encodedSample)<self.maxSampleLength: # add PAD (index is length of vocab)
                encodedSample.append(self.oneEncoderLength)
            if self.verbose:
                print(seq)
                print(encodedSample)
            return encodedSample
        
        if self.verbose:
            print('Raw Data')
            for k in proteinDict:
                print('--------',k,'--------')
                print(proteinDict[k])
            print('Transformed Sample -------')
            print('Seq',seq)
            print('Existence', existence)
            print('KWs',kws)
            print('Taxa',taxa)
            
        encodedSample = []
        if self.oneEncoderLength:
            if justidx:
                for tax_line in taxa:
                    encodedSample.extend([self.taxa_to_ctrl[tax] for tax in tax_line])
                for kw_line in kws:
                    encodedSample.extend([self.kw_to_ctrl[kw] for kw in kw_line])
                seq_idx = 0
                while (len(encodedSample)<self.maxSampleLength) and (seq_idx<len(seq)):
                    encodedSample.append(self.aa_to_ctrl[seq[seq_idx]])
                    seq_idx += 1
                thePadIndex = len(encodedSample)
                while len(encodedSample)<self.maxSampleLength: # add PAD (index is length of vocab)
                    encodedSample.append(self.oneEncoderLength)
            else: # TODO: increase dim by 1 and include padding token. OUTDATED code below
                for tax in taxa:
                    token = np.zeros(self.oneEncoderLength, dtype = np.uint8)
                    token[self.taxa_to_ctrl[tax]] = 1
                    encodedSample.append(token)
                for kw in kws:
                    token = np.zeros(self.oneEncoderLength, dtype = np.uint8)
                    token[self.kw_to_ctrl[kw]] = 1
                    encodedSample.append(token)
                seq_idx = 0
                while (len(encodedSample)<self.maxSampleLength) and (seq_idx<len(seq)):
                    token = np.zeros(self.oneEncoderLength, dtype = np.uint8)
                    token[self.aa_to_ctrl[seq[seq_idx]]] = 1
                    encodedSample.append(token)
                    seq_idx += 1
                # Padding
                while len(encodedSample)<self.maxSampleLength:
                    token = np.zeros(self.oneEncoderLength, dtype = np.uint8)
                    encodedSample.append(token)

        return encodedSample, existence, thePadIndex

if __name__ == "__main__":
    chunknum = 0
    with open('/export/share/amadani/protein-data/train_test/train'+str(chunknum)+'.p','rb') as handle:
        train_chunk = pickle.load(handle)
    uid = 'UPI000000BF1A'
    #uid = random.sample(train_chunk.keys(),1)[0]
    obj = transformProtein(verbose=True, dropRate = 0.0, maxTaxaPerSample = 3, maxKwPerSample = 5, 
                           selectSwiss = 1.0, selectTrembl = 1.0, seqonly = True)
    print(obj.transformSample(train_chunk[uid]))
