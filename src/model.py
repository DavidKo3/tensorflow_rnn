import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq

import numpy as np

class Model():
    def __init__(self, args, infer=False):
        self.args= args
        if infer:
            self.batch_size = 1
            args.seq_length = 1
        
        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {} ".format(args.model))
        
        cell = cell_fn(args.rnn_size, state_is_tuple=True)
        
        self.cell = cell = rnn_cell.MultiRNNCell([cell]*args.num_layers, state_is_tuple=True)