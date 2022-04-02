import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm
import math, copy, time
from torch.autograd import Variable

class EncoderDecoder(nn.Module):
    def __init__(self, query_encoder, vid_graph_encoder, decoder, query_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        # emb
        self.query_embed = query_embed
        self.tgt_embed = tgt_embed

        # encoder
        self.query_encoder = query_encoder
        self.vid_graph_encoder = vid_graph_encoder

        # decoder
        self.decoder = decoder
        self.generator = generator

    def forward(self, src, src_mask, fts, trg, trg_mask):
        encoded_output = self.encode(src, src_mask, fts)
        return self.decode(encoded_output, src_mask, trg, trg_mask)

    def vid_encode(self, video_features, a_vid):
        output = self.vid_graph_encoder(video_features, a_vid)
        return output

    def encode(self, query, query_mask, vid=None):

        # vid[1] 表示 ["bbox_features"]中特征(batch_size, 32, 20, 256)  vid[0] 是对graph的表示　(batch_size, 32, 20, 20)
        vid_graph_output = self.vid_encode(vid[1], vid[0])     # vid_graph_output.shape 应为 (batch_size, 32, 512)

        output = self.query_encoder(self.query_embed(query), query_mask, vid_graph_output)
        return output

    def decode(self, query_memory, query_mask, tgt, tgt_mask):
        encoded_tgt = self.tgt_embed(tgt)
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

    def forward(self, x, mask, vid_fts):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask, vid_fts)
        return self.norm(x)


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
    def __init__(self, size, self_attn, vid_self_attn, vid_attn, vid_ff, ff1, dropout, src_attn=None):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.vid_attn = vid_attn
        self.vid_self_attn = vid_self_attn
        self.vid_ff = vid_ff
        self.ff1 = ff1
        if self.src_attn:
            self.sublayer = clones(SublayerConnection(size, dropout), 6)
        else:
            self.sublayer = clones(SublayerConnection(size, dropout), 4)
        self.size = size

    def forward(self, seq, seq_mask, vid_ft):
        count = 0
        # vid_ft = self.sublayer[count](vid_ft, lambda vid_ft: self.vid_self_attn(vid_ft, vid_ft, vid_ft))
        # count += 1
        vid_ft = self.sublayer[count](vid_ft, self.vid_ff)
        count += 1

        seq = self.sublayer[count](seq, lambda seq: self.self_attn(seq, seq, seq, seq_mask))
        count += 1
        seq = self.sublayer[count](seq, self.ff1)

        vid_seq = self.sublayer[count](seq, lambda seq: self.vid_attn(seq, vid_ft, vid_ft))    # torch.Size([2, 40, 512]) 即视频给文本上attention
        count += 1

        return vid_seq


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
    if time_weighting is not None:
        p_attn = p_attn * time_weighting[:, :T, :T]

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
        # self.time_shift = nn.ZeroPad2d((0, 0, 1, 0))
        # self.time_weighting = nn.Parameter(torch.ones(self.h, 128, 128))

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
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
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


"""Graph"""


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, n_layers):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, input_feature, adj):
        lv = 0
        cur_message_layer = input_feature

        while lv < self.n_layers:
            n2npool = torch.matmul(adj, cur_message_layer)
            cur_message_layer = torch.matmul(n2npool, self.W)
            lv += 1

        # feature change
        video_spatial = cur_message_layer
        video_spatial_feature = torch.mean(video_spatial, dim=-2, keepdim=True)
        video_spatial_feature = torch.squeeze(video_spatial_feature)

        return video_spatial_feature


class VGraphEncoder(nn.Module):

    def __init__(self, g_in_fea, g_out_fea, g_layers, d_model, dropout):
        super(VGraphEncoder, self).__init__()
        self.g_in_fea = g_in_fea
        self.g_out_fea = g_out_fea
        self.g_layers = g_layers
        self.d_model = d_model

        self.gcn = GraphConvolution(g_in_fea, g_out_fea, g_layers)
        self.linear = nn.Linear(g_out_fea, d_model)
        self.relu = nn.LeakyReLU()
        self.pe = PositionalEncoding(d_model, dropout)

    def forward(self, input_feature, adj):
        v = self.gcn(input_feature, adj)
        v = self.relu(self.linear(v))  # 2layer
        # 20201018 因为不要relu和linear时，结果不好
        # v = self.relu(v)
        v_output = self.pe(v)

        return v_output


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1, GCN_layer=3, co_attn=False):

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    generator = Generator(d_model, tgt_vocab)
    query_embed = [Embeddings(d_model, src_vocab), c(position)]
    tgt_embed = [Embeddings(d_model, tgt_vocab), c(position)]
    query_embed = nn.Sequential(*query_embed)
    tgt_embed = nn.Sequential(*tgt_embed)


    """graph"""
    vid_graph_encoder = VGraphEncoder(556, 556, GCN_layer, d_model, dropout)
    vid_self_attn = c(attn)
    vid_attn = c(attn)

    if co_attn:
        query_encoder = Encoder(EncoderLayer(d_model, c(attn), vid_self_attn, vid_attn, c(ff), c(ff), dropout, c(attn)), N)
    else:
        query_encoder = Encoder(EncoderLayer(d_model, c(attn), vid_self_attn, vid_attn, c(ff), c(ff), dropout), N)

    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)

    model = EncoderDecoder(
        query_encoder=query_encoder,
        vid_graph_encoder=vid_graph_encoder,
        decoder=decoder,
        query_embed=query_embed,
        tgt_embed=tgt_embed,
        generator=generator,)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return model
