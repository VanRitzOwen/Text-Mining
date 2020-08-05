import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import const


class Bilstm(torch.nn.Module):
    def __init__(self, batch_size, output_size, hidden_size, vocab_size, embed_dim, bidirectional, dropout, attention_size, sequence_length):
        super(Bilstm, self).__init__()

        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.sequence_length = sequence_length
        self.lookup_table = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=const.PAD)
        self.lookup_table.weight.data.uniform_(-1., 1.)

        self.layer_size = 1
        self.lstm = nn.LSTM(self.embed_dim,
                            self.hidden_size,
                            self.layer_size,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.layer_size = self.layer_size * 2
        else:
            self.layer_size = self.layer_size

        self.attention_size = attention_size

        self.w_omega = Variable(torch.zeros(self.hidden_size * self.layer_size, self.attention_size))
        self.u_omega = Variable(torch.zeros(self.attention_size))

        self.label = nn.Linear(hidden_size * self.layer_size, output_size)

    def attention_net(self, lstm_output):

        output_reshape = torch.Tensor.reshape(lstm_output, [-1, self.hidden_size*self.layer_size])
        attn_tanh = torch.tanh(torch.mm(output_reshape, self.w_omega))
        attn_hidden_layer = torch.mm(attn_tanh, torch.Tensor.reshape(self.u_omega, [-1, 1]))
        exps = torch.Tensor.reshape(torch.exp(attn_hidden_layer), [-1, self.sequence_length])
        alphas = exps / torch.Tensor.reshape(torch.sum(exps, 1), [-1, 1])
        alphas_reshape = torch.Tensor.reshape(alphas, [-1, self.sequence_length, 1])
        state = lstm_output.permute(1, 0, 2)
        attn_output = torch.sum(state * alphas_reshape, 1)

        return attn_output

    def forward(self, input_sentences, batch_size=None):
        input = self.lookup_table(input_sentences)
        input = input.permute(1, 0, 2)

        h_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))
        c_0 = Variable(torch.zeros(self.layer_size, self.batch_size, self.hidden_size))

        lstm_output, (final_hidden_state, final_cell_state) = self.lstm(input, (h_0, c_0))
        attn_output = self.attention_net(lstm_output)
        logits = self.label(attn_output)
        return logits

