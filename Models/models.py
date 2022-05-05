# -*- coding: utf-8 -*-
# @Time : 2022/4/24 20:59
# @Author : Bingshuai Liu
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, encoder_hidden_dim, decoder_hidden_dim, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_dim, encoder_hidden_dim)
        self.fc = nn.Linear(encoder_hidden_dim * 2, decoder_hidden_dim)
        self.rnn = nn.GRU(emb_dim, encoder_hidden_dim, bidirectional=True)

    def forward(self, source):
        # source: [source len, batch size]
        embedded = self.dropout(self.embedding(source))

        # outputs: [source len, batch size, encoder hidden dim * num directions]
        # hidden = [n layers * num directions, batch size, encoder hidden dim]
        outputs, hidden = self.rnn(embedded)

        forwards_rnn = hidden[-2, :, :]
        backwards_rnn = hidden[-1, :, :]
        # 将前向和反向的RNN拼接到一起
        ffn = self.fc(torch.cat((forwards_rnn, backwards_rnn), dim=1))

        hidden = torch.tanh(ffn)

        # outputs : [source len, batch size, encoder hidden dim * 2]
        # hidden : [batch size, decoder hidden dim]
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        return

    def forward(self):
        return


class DHEA(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.device = device
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, tfr=0.5):
        # source [source length, batch size]
        # target [target length, batch size]
        target_len = target.shape[0]
        batch_size = source.shape[1]
        target_vocab_size = self.decoder.output_dim

        encoder_outputs, hidden = self.encoder(source)

        # 初始化输出,用于存储每个时间步的结果
        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)


        # for t in range(1, target_len):
            # output, hidden = self.decoder()
        return
