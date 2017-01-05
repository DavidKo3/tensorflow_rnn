"""

https://github.com/sherjilozair/char-rnn-tensorflow/blob/master/utils.py
"""
# -*- coding: cp949 -*-
import codecs
import os
import collections
from six.moves import cPickle
import numpy as np

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding='utf-8'):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.encoding = encoding
    
        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")
        
        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocesssed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()
        
    def preprocess(self, input_file, vocab_file, tensor_file):    
        with codecs.open(input_file, "r", encoding=self.encoding) as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items() , key=lambda x: -x[1]) # sorting values in order of decreasing value
        self.chars, _ = zip(*count_pairs)   # *args : variable array
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)
    
    def load_preprocess(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = cPickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size/ (self.batch_size * self.seq_length))
    
    def create_batches(self):
        self.num_batches = int(self.tensor.size/ (self.batch_size * self.seq_length))
        
        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches == 0:
            assert False , "Not enough data. Make seq_length and batch_size small."
        
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        
        
a= np.array([1,2,3,4,5])

b= a[:3]

c= a[2:]

print (b)

        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    