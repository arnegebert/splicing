import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
import copy
import torch
import numpy as np
import math
from torch.autograd import Variable


class MNISTModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class NaivePSIModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=(1,3), padding=0)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1,3), padding=0)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(156*16, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = x.view(-1, 4, 1, 2*80)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2_drop(self.conv2(x)))
        x = x.view(-1, 156*16)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return torch.sigmoid(x)


class DeepSplicingCodeSmoll(BaseModel):

    def __init__(self):
        super().__init__()

        self.fc_in = 18*8*2
        self.conv1_start = nn.Conv1d(4, 32, kernel_size=3)
        self.conv1_drop_start = nn.Dropout2d(0.2)

        self.conv2_start = nn.Conv1d(32, 8, kernel_size=3)
        self.conv2_drop_start = nn.Dropout2d(0.2)

        self.conv3_start = nn.Conv1d(8, 8, kernel_size=3)
        self.conv3_drop_start = nn.Dropout2d(0.2)

        # end conv blocks here...
        self.conv1_end = nn.Conv1d(4, 32, kernel_size=3)
        self.conv1_drop_end = nn.Dropout2d(0.2)

        self.conv2_end = nn.Conv1d(32, 8, kernel_size=3)
        self.conv2_drop_end = nn.Dropout2d(0.2)

        self.conv3_end = nn.Conv1d(8, 8, kernel_size=3)
        self.conv3_drop_end = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(self.fc_in, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)


    def forward(self, data):
        # [128, 80, 4]
        start, end = data[:, 0], data[:, 1]
        start, end = start.view(-1, 4, 80), end.view(-1, 4, 80)
        x = F.relu(self.conv1_drop_start(self.conv1_start(start)))

#        x = F.max_pool1d(F.relu(self.conv1_drop_start(self.conv1_start(start))), 2)
        x = F.max_pool1d(F.relu(self.conv2_drop_start(self.conv2_start(x))), 2)
        x = F.max_pool1d(F.relu(self.conv3_drop_start(self.conv3_start(x))), 2)

        # xx = F.max_pool1d(F.relu(self.conv1_drop_end(self.conv1_end(end))), 2)
        xx = F.relu(self.conv1_drop_end(self.conv1_end(end)))
        xx = F.max_pool1d(F.relu(self.conv2_drop_end(self.conv2_end(xx))), 2)
        xx = F.max_pool1d(F.relu(self.conv3_drop_end(self.conv3_end(xx))), 2)

        feats = torch.cat((x, xx), dim=1)
        feats = feats.view(-1, self.fc_in)
        y = self.drop_fc(F.relu(self.fc1(feats)))
        y = torch.sigmoid(self.fc2(y))
        return y


class DSC(BaseModel):

    def __init__(self):
        super().__init__()

        self.fc_in = 15*8*2 + 3
        self.conv1_start = nn.Conv1d(4, 32, kernel_size=7)
        self.conv1_drop_start = nn.Dropout2d(0.2)

        self.conv2_start = nn.Conv1d(32, 8, kernel_size=4)
        self.conv2_drop_start = nn.Dropout2d(0.2)

        self.conv3_start = nn.Conv1d(8, 8, kernel_size=3)
        self.conv3_drop_start = nn.Dropout2d(0.2)

        # end conv blocks here...
        self.conv1_end = nn.Conv1d(4, 32, kernel_size=7)
        self.conv1_drop_end = nn.Dropout2d(0.2)

        self.conv2_end = nn.Conv1d(32, 8, kernel_size=4)
        self.conv2_drop_end = nn.Dropout2d(0.2)

        self.conv3_end = nn.Conv1d(8, 8, kernel_size=3)
        self.conv3_drop_end = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(self.fc_in, 64)
        self.drop_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, seqs, lens):
        # [128, 142, 4]
        start, end = seqs[:, 0], seqs[:, 1]
        start, end = start.view(-1, 4, 142), end.view(-1, 4, 142)

        x = F.max_pool1d(F.relu(self.conv1_drop_start(self.conv1_start(start))), 2)
        x = F.max_pool1d(F.relu(self.conv2_drop_start(self.conv2_start(x))), 2)
        x = F.max_pool1d(F.relu(self.conv3_drop_start(self.conv3_start(x))), 2)

        xx = F.max_pool1d(F.relu(self.conv1_drop_end(self.conv1_end(end))), 2)
        xx = F.max_pool1d(F.relu(self.conv2_drop_end(self.conv2_end(xx))), 2)
        xx = F.max_pool1d(F.relu(self.conv3_drop_end(self.conv3_end(xx))), 2)

        feats = torch.cat((x, xx), dim=1)
        feats = feats.view(-1, self.fc_in-3)
        feats = torch.cat((feats, lens), dim=1)
        y = self.drop_fc(F.relu(self.fc1(feats)))
        y = torch.sigmoid(self.fc2(y))

        return y


class DSCKiller(BaseModel):

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 1)

    def forward(self, seqs, lens):
        y = torch.sigmoid(self.lin(lens))
        return y

class DSCKillerKiller(BaseModel):

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1, bias=True)

    def forward(self, seqs, lens):
        y = torch.sigmoid(self.lin(lens[:, 1].view(-1, 1)))
        return y


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """Take in and process masked src and target sequences."""
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
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
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same size."""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """Follow Figure 1 (left) for connections."""
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """Decoder is made of self-attn, src-attn, and feed forward (defined below)"""

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """Follow Figure 1 (right) for connections."""
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        # Fills elements of self tensor with value where mask is True.
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # First linear layer works on query, second on key, third on value
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
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
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


def make_model(src_vocab, tgt_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1):
    """Helper: Construct a model from hyperparameters."""
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn),
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

# Small example model.
# tmp_model = make_model(10, 10, 2)