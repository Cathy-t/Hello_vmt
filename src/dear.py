# -*- coding: utf-8 -*-

# @Time : 2021/5/23 18:57
# @Author : Cathy
# @FileName: dear.py
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable

"""act+transformer+circle"""

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, vid_encoder, decoder, en_embed, zh_embed, en_generator, zh_generator, share_classifier):
        super(EncoderDecoder, self).__init__()
        # emb
        self.en_embed = en_embed
        self.zh_embed = zh_embed

        # encoder-decoder 主网络，需要被共享参数的位置
        self.encoder = encoder
        self.vid_encoder = vid_encoder
        self.decoder = decoder

        # generator
        self.en_generator = en_generator
        self.zh_generator = zh_generator

        # act label classifier
        # self.en_act_pred = None
        # self.zh_act_pred = None


    def forward(self, en, ensrc_mask, entgt_mask, zh, zhsrc_mask, zhtgt_mask, video, video_mask):
        en_encoded, en_act_pred = self.encode(en, ensrc_mask, video, video_mask)
        en_decoded = self.decode(en_encoded, ensrc_mask, zh[:, :-1], zhtgt_mask)

        zh_encoded, zh_act_pred = self.encode(zh, zhsrc_mask, video, video_mask, type='zh2en')
        zh_decoded = self.decode(zh_encoded, zhsrc_mask, en[:, :-1], entgt_mask, type='zh2en')
        return en_decoded, zh_decoded, en_act_pred, zh_act_pred

    def vid_encode(self, video_features):
        output = self.vid_encoder(video_features)
        return output

    def encode(self, query, query_mask, video, video_mask, type='en2zh'):
        if type == 'en2zh':
            output = self.encoder(self.en_embed(query), query_mask, self.vid_encode(video), video_mask, type)
        else:
            output = self.encoder(self.zh_embed(query), query_mask, self.vid_encode(video), video_mask, type)
        return output

    def decode(self, query_memory, query_mask, tgt, tgt_mask, type='en2zh'):
        if type == 'en2zh':
            encoded_tgt = self.zh_embed(tgt)
        else:
            encoded_tgt = self.en_embed(tgt)
        return self.decoder(encoded_tgt, query_memory, query_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

    def inference(self, x):
        return self.proj(x)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, video, video_mask, type):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x, act_pred = layer(x, mask, video, video_mask, type)
        return self.norm(x), act_pred


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size"
        return x + self.dropout(sublayer(self.norm(x)))

    def expand_forward(self, x, sublayer):
        out = self.dropout(sublayer(self.norm(x)))
        out = out.mean(1).unsqueeze(1).expand_as(x)
        return x + out

    def nosum_forward(self, x, sublayer):
        return self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, vid_attn, ff1, dropout, seq_attn=None, ff2=None, classifier=None):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.vid_attn = vid_attn
        self.ff1 = ff1
        self.sublayer = clones(SublayerConnection(size, dropout), 5)
        self.seq_attn = seq_attn
        self.ff2 = ff2
        self.size = size

        self.classifier = classifier

    def forward(self, seq, seq_mask, video, video_mask, type):
        seq = self.sublayer[0](seq, lambda seq: self.self_attn(seq, seq, seq, seq_mask))
        vid_seq = self.sublayer[1](seq, lambda seq: self.vid_attn(seq, video, video, video_mask))

        seq_vid = self.sublayer[2](video, lambda video: self.seq_attn(video, seq, seq, seq_mask))
        seq_vid = self.sublayer[3](seq_vid, self.ff2)

        if isinstance(self.classifier, dict):
            if type == 'en2zh':
                act_pred = F.log_softmax(self.classifier['en'](seq_vid), dim=-1)
            else:
                act_pred = F.log_softmax(self.classifier['zh'](seq_vid), dim=-1)
        else:
            act_pred = F.log_softmax(self.classifier(seq_vid))

        return self.sublayer[4](vid_seq, self.ff1), act_pred


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, query_memory, query_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, query_memory, query_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, q_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = q_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, q_memory, q_mask, tgt_mask):
        count = 0
        x = self.sublayer[count](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        count += 1
        x = self.sublayer[count](x, lambda x: self.src_attn(x, q_memory, q_memory, q_mask))
        count += 1
        return self.sublayer[count](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None, time_weighting=None, T=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    # time_weighting
    # p_attn = p_attn * time_weighting[:, :T, :T]

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, d_in=-1, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.d_model = d_model
        self.h = h
        if d_in < 0:
            d_in = d_model
        self.linears = clones(nn.Linear(d_in, d_model), 3)
        self.linears.append(nn.Linear(d_model, d_in))
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        # trick
        self.time_shift = nn.ZeroPad2d((0, 0, 1, 0))
        self.time_weighting = nn.Parameter(torch.ones(self.h, 128, 128))

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches, qT, qC = query.size()
        # _, kT, kC = key.size()
        # _, vT, vC = value.size()

        #  trick:  time-mixing
        # query = torch.cat([self.time_shift(query)[:, :qT, :qC//2], query[:, :qT, qC//2:]], dim=2)
        # key = torch.cat([self.time_shift(key)[:, :kT, :kC//2], key[:, :kT, kC//2:]], dim=2)
        # value = torch.cat([self.time_shift(value)[:, :vT, :vC // 2], value[:, :vT, vC // 2:]], dim=2)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # trick : 加入　qT　和　time-weighting
        # x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout, time_weighting=self.time_weighting, T=qT)
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1, d_out=-1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        if d_out < 0:
            d_out = d_model
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        try:
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        except:
            x = x.unsqueeze(0)
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def make_model(en_vocab, zh_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, share_classifier=False):

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    en_generator = Generator(d_model, en_vocab)
    zh_generator = Generator(d_model, zh_vocab)

    en_embed = [Embeddings(d_model, en_vocab), c(position)]
    en_embed = nn.Sequential(*en_embed)

    zh_embed = [Embeddings(d_model, zh_vocab), c(position)]
    zh_embed = nn.Sequential(*zh_embed)

    # 构建分类器
    if share_classifier:
        classifier = nn.Linear(d_model, 401, bias=False)
    else:
        en_classifier = nn.Linear(d_model, 401, bias=False).cuda()
        zh_classifier = nn.Linear(d_model, 401, bias=False).cuda()
        classifier = {'en': en_classifier, 'zh': zh_classifier}

    encoder = Encoder(EncoderLayer(d_model, c(attn), c(attn), c(ff), dropout, c(attn), c(ff), classifier), N)

    ff_layers = [nn.Linear(400, d_model), nn.ReLU(), c(position)]
    vid_encoder = nn.Sequential(*ff_layers)

    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)

    model = EncoderDecoder(
        encoder=encoder,
        vid_encoder=vid_encoder,
        decoder=decoder,
        en_embed=en_embed,
        zh_embed=zh_embed,
        en_generator=en_generator,
        zh_generator=zh_generator,
        share_classifier=share_classifier)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model

