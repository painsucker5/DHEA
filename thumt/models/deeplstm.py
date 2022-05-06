# -*- coding: utf-8 -*-
# @Time : 2022/5/4 11:07
# @Author : Bingshuai Liu
import torch.nn as nn
import torch
import thumt.utils as utils
from thumt.modules.feed_forward import FeedForward
import thumt.utils as utils
import thumt.modules as modules


# class Encoder(nn.Module):
#     def __init__(self):
#
#     def forward(self):
#         return ""

class LSTM(modules.Module):
    def __init__(self, input_size, output_size, lstm_normalization, lstm_activation):
        super(LSTM, self).__init__()
        self.lstmCell = modules.LSTMCell(input_size, output_size,
                                         normalization=lstm_normalization,
                                         activation=lstm_activation)

    def forward(self, input, init_state=None, state=None):
        # 将数据
        # input: [batch, length, hidden_size] -> [length, batch, hidden_size]
        input = torch.transpose(input, 0, 1)

        t = input.size(0)
        batch = input.size(1)

        # 初始化lstm_state
        if state is None:
            lstm_state = self.lstmCell.init_state(batch, dtype=input.dtype, device=input.device)
        else:
            lstm_state = state['lstm_state']

        hiddens = []
        for i in range(t):
            hidden, lstm_state = self.lstmCell(input[i], lstm_state)

            # 在第一位前增加一个维度
            hidden = torch.unsqueeze(hidden, dim=0)
            hiddens.append(hidden)
        hiddens = torch.cat(hiddens, dim=0)

        # 与最开始的操作对应, 将数据还原到原来的格式
        # hiddens: [length, batch, hidden_size] -> [batch, length, hidden_size]
        hiddens = torch.transpose(hiddens, 0, 1)
        return hiddens, lstm_state


class EncoderLayer(modules.Module):
    def __init__(self, params):
        super(EncoderLayer, self).__init__()
        self.dropout = params.residual_dropout
        self.lstm = LSTM(params.hidden_size, params.hidden_size, )
        self.ffn = modules.FeedForward(params.hidden_size, params.filter_size,
                                       dropout=params.relu_dropout)

    def forward(self, input, init_state=None, batch_first=True):
        hiddens, lstm_state = self.lstm(input, init_state, None, batch_first)
        hiddens = self.ffn(hiddens)
        hiddens = nn.functional.dropout(hiddens, self.dropout, self.training)
        return hiddens, lstm_state


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        # self.encoder_layers = params.num_encoder_layers
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(params) for i in range(params.num_encoder_layers)
        ])

    def forward(self, input, batch_first=True):
        init_state = None
        for i in range(len(self.encoder_layers)):
            if i % 2 == 0:
                # 奇数层 正向LSTM
                # input : [batch, length ,hidden_size]
                input, state = self.encoder_layers[i](input, init_state, batch_first)
            else:
                # 偶数层 反向LSTM
                input, state = self.encoder_layers[i](torch.flip(input, dims=[1]), init_state, batch_first)
                input = torch.flip(input, dims=[1])
        return input


class DecoderLayer(modules.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()

    def forward(self):
        return


class Decoder(modules.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList([
            Decoder(params) for i in range(params.num_decoder_layers)
        ])

        return

    def forward(self):
        return


class DeepLSTM(modules.Module):
    def __init__(self, params, name="deeplstm"):

        super(DeepLSTM, self).__init__()
        self.params = params
        # self.decoder = Decoder(params)
        self.encoder = Encoder(params)
        self.build_embedding(params)
        self.reset_parameters()
        self.hidden_size = params.hidden_size
        self.num_decoder_layers = params.num_decoder_layers
        self.num_encoder_layers = params.num_encoder_layers

    def build_embedding(self, params):
        svoc_size = len(params.vocabulary["source"])
        tvoc_size = len(params.vocabulary["target"])

        # if params.shared_source_target_embedding and svoc_size != tvoc_size:
        #     raise ValueError("Cannot share source and target embedding.")

        # if not params.shared_embedding_and_softmax_weights:
        self.softmax_weights = torch.nn.Parameter(
            torch.empty([tvoc_size, params.hidden_size]))
        self.add_name(self.softmax_weights, "softmax_weights")

        # if not params.shared_source_target_embedding:
        self.source_embedding = torch.nn.Parameter(
            torch.empty([svoc_size, params.hidden_size]))
        self.target_embedding = torch.nn.Parameter(
            torch.empty([tvoc_size, params.hidden_size]))
        self.add_name(self.source_embedding, "source_embedding")
        self.add_name(self.target_embedding, "target_embedding")
        # else:
        #     self.weights = torch.nn.Parameter(
        #         torch.empty([svoc_size, params.hidden_size]))
        #     self.add_name(self.weights, "weights")

        self.bias = torch.nn.Parameter(torch.zeros([params.hidden_size]))
        self.add_name(self.bias, "bias")

    def encode(self, features, state):
        source_sequence = features["source"]
        src_mask = features["source_mask"]
        enc_attn_bias = self.masking_bias(src_mask)

        inputs = torch.nn.functional.embedding(source_sequence, self.source_embedding)
        inputs = inputs * (self.hidden_size ** 0.5)
        inputs = inputs + self.bias

        # 开始encode
        encoder_output = self.encoder(inputs)
        enc_attn_bias = enc_attn_bias.to(inputs)

        state["encoder_output"] = encoder_output
        state["enc_attn_bias"] = enc_attn_bias

        return state

    def decode(self, features, state, mode="infer"):
        tgt_seq = features["target"]

        enc_attn_bias = state["enc_attn_bias"]
        dec_attn_bias = self.causal_bias(tgt_seq.shape[1])
        # target embedding
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

        # 准备进入decoder
        decoder_output = self.decoder(decoder_input, dec_attn_bias,
                                      enc_attn_bias, encoder_output, state)

        decoder_output = torch.reshape(decoder_output, [-1, self.hidden_size])
        decoder_output = torch.transpose(decoder_output, -1, -2)
        logits = torch.matmul(self.softmax_embedding, decoder_output)
        logits = torch.transpose(logits, 0, 1)

        return logits, state

    def forward(self, features, labels):
        # [batch_size, length]
        mask = features["target_mask"]

        state = self.empty_state(features["target"].shape[0], labels.device)
        state = self.encode(features, state)

        # source_sequence = features['source']
        # target_sequence = features['target']
        # source_length = source_sequence.size(0)
        # source_bs = source_sequence.size(1)

        # state = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        # lstm = []
        # for i in range(source_length):
        #     print(source_sequence[i].size())
        # output, hidden = self.lstm(source_sequence[i])

        return state

    @staticmethod
    def masking_bias(mask, inf=-1e9):
        ret = (1.0 - mask) * inf
        return torch.unsqueeze(torch.unsqueeze(ret, 1), 1)

    @staticmethod
    def causal_bias(length, inf=-1e9):
        ret = torch.ones([length, length]) * inf
        ret = torch.triu(ret, diagonal=1)
        return torch.reshape(ret, [1, 1, length, length])

    def empty_state(self, batch_size, device):
        state = {
            "decoder": {
                "layer_%d" % i: {
                    "k": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device),
                    "v": torch.zeros([batch_size, 0, self.hidden_size],
                                     device=device),
                    "lstm_state": (torch.zeros([batch_size, self.hidden_size], device=device),
                                   torch.zeros([batch_size, self.hidden_size], device=device))
                } for i in range(self.num_decoder_layers)
            }
        }

    def reset_parameters(self):
        nn.init.normal_(self.source_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)
        nn.init.normal_(self.target_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)
        nn.init.normal_(self.source_embedding, mean=0.0,
                        std=self.params.hidden_size ** -0.5)

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
            # normalization="after",
            # shared_embedding_and_softmax_weights=False,
            # shared_source_target_embedding=False,
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
            clip_grad_norm=0.0,
            lstm_normalization=False,
            lstm_activation=None,
        )

        return params

    def default_params(name=None):
        if name == "base":
            return DeepLSTM.base_params()
        else:
            return None
