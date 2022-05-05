# -*- coding: utf-8 -*-
# @Time : 2022/5/4 11:07
# @Author : Bingshuai Liu
import torch.nn as nn
import torch
import thumt.utils as utils
from thumt.modules.feed_forward import FeedForward

# class Encoder(nn.Module):
#     def __init__(self):
#
#     def forward(self):
#         return ""

class LSTM(nn.Module):
    def __init__(self,params):
        super(LSTM, self).__init__()

    def forward(self):
        return

class EncoderLayer(nn.Module):
    def __init__(self,params):
        super(EncoderLayer, self).__init__()
        self.lstm = LSTM(params)
        # self.ffn =

    def forward(self):
        return

class Encoder(nn.Module):
    def __init__(self,params):
        super(Encoder, self).__init__()
        self.encoder_layers = params.num_encoder_layers
        # self.lstm = LSTM(params.hidden_size,params)

    def forward(self, input):
        for i in range(self.encoder_layers):
            if i % 2 == 0:
                # 反向LSTM
                return
            else:
                # 正向LSTM
                return
        return
class DeepLSTM(nn.Module):
    def __init__(self, params, name="deeplstm"):

        super(DeepLSTM, self).__init__()
        self.encoder = Encoder(params)
        # self.lstm = nn.LSTM(param.hidden_size, param.hidden_size)
        # self.ffn = nn.Linear(pa)

    def encode(self, features, state):
        src_seq = features["source"]
        src_mask = features["source_mask"]
        enc_attn_bias = self.masking_bias(src_mask)

        inputs = torch.nn.functional.embedding(src_seq, self.src_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.bias
        inputs = nn.functional.dropout(self.encoding(inputs), self.dropout,
                                       self.training)

        enc_attn_bias = enc_attn_bias.to(inputs)
        encoder_output = self.encoder(inputs, enc_attn_bias)

        state["encoder_output"] = encoder_output
        state["enc_attn_bias"] = enc_attn_bias

        return state

    def decode(self, features, state, mode="infer"):
        tgt_seq = features["target"]

        enc_attn_bias = state["enc_attn_bias"]
        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])

        targets = torch.nn.functional.embedding(tgt_seq, self.tgt_embedding)
        targets = targets * (self.hidden_size ** 0.5)

        decoder_input = torch.cat(
            [targets.new_zeros([targets.shape[0], 1, targets.shape[-1]]),
             targets[:, 1:, :]], dim=1)
        decoder_input = nn.functional.dropout(self.encoding(decoder_input),
                                              self.dropout, self.training)

        encoder_output = state["encoder_output"]
        dec_attn_bias = dec_attn_bias.to(targets)

        if mode == "infer":
            decoder_input = decoder_input[:, -1:, :]
            dec_attn_bias = dec_attn_bias[:, :, -1:, :]

        decoder_output = self.decoder(decoder_input, dec_attn_bias,
                                      enc_attn_bias, encoder_output, state)

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_embedding, decoder_output)
        logits = torch.transpose(logits, 0, 1)

        return logits, state



    def forward(self, features, labels):
        # [length, batch_size]
        source_sequence = features['source']
        target_sequence = features['target']
        source_length = source_sequence.size(0)
        source_bs = source_sequence.size(1)

        # state = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        lstm = []
        for i in range(source_length):
            print(source_sequence[i].size())
            # output, hidden = self.lstm(source_sequence[i])

        return source_sequence

    @staticmethod
    def base_params():
        params = utils.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=4,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            normalization="after",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            warmup_steps=4000,
            train_steps=100000,
            learning_rate=7e-4,
            learning_rate_schedule="linear_warmup_rsqrt_decay",
            batch_size=4096,
            fixed_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0
        )

        return params

    def default_params(name=None):
        if name == "base":
            return DeepLSTM.base_params()
        else:
            return None
