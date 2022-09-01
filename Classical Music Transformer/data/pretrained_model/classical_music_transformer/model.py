import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math

def strided_axis1(a, window, hop):
    # Zero-padding
    npad = (a.shape[1] - window) % hop + 1
    if npad != 0 and hop != 1:
        b = np.lib.pad(a, ((0, 0), (0,npad)), 'constant', constant_values=0)
    else:
        b = np.array(a)

    # Length of 3D output array along its axis=1
    nd1 = int((b.shape[1] - window)/hop) + 1

    # Store shape and strides info
    m, n = b.shape
    s0, s1 = b.strides

    # Finally use strides to get the 3D array view
    return np.lib.stride_tricks.as_strided(b, shape=(nd1, m, window), strides=(s1*hop, s0, s1))

def create_mask(src):
    size = src.size(1) # get seq_len for matrix
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask =  Variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.cuda()
    src_mask = np_mask
    
    return src_mask

def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0):
        super(MultiHeadAttention, self).__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        q = self.k_linear(q).view(batch_size, -1, self.h, self.d_k)
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k)
        v = self.k_linear(v).view(batch_size, -1, self.h, self.d_k)
        
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        concat = scores.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out(concat)
        
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, dropout = 0):
        super().__init__() 
        self.d_ff = 4*d_model
        self.linear_1 = nn.Linear(d_model, self.d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(self.d_ff, d_model)
        
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
        
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm
    
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        div_term_even = 1/10000 ** (2 * torch.arange(0, d_model, 2).float() / d_model)
        div_term_odd = 1/10000 ** (2 * (torch.arange(0, d_model, 2).float() + 1) / d_model)
        pe[:, 0::2] = torch.sin(div_term_even)
        pe[:, 1::2] = torch.cos(div_term_odd)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout = 0):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm_2 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = MultiHeadAttention(heads, d_model)
        self.ff = FeedForward(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super().__init__()
        self.N = N
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.norm = Norm(d_model)
        
    def forward(self, src, mask):
        x = self.pe(src)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self, d_type=5, d_barpos=34, d_tone=25, d_chord=109, d_dur=65, d_pitch=129, d_attention=512, N=12, heads=8):
        super().__init__()
        self.encoder = Encoder(d_attention, N, heads)

        self.intype = nn.Linear(d_type, 32)
        self.inbarpos = nn.Linear(d_barpos, 64)
        self.intone = nn.Linear(d_tone, 64)
        self.inchord = nn.Linear(d_chord, 256)
        self.indur = nn.Linear(d_dur, 256)
        self.inpitch = nn.Linear(d_pitch, 512)
        self.inlinear = nn.Linear(32+64+64+256+256+512, d_attention)
        
        self.outtype = nn.Linear(d_attention, d_type)
        self.outconcat = nn.Linear(d_attention + 32, d_attention)
        self.outbarpos = nn.Linear(d_attention, d_barpos)
        self.outtone = nn.Linear(d_attention, d_tone)
        self.outchord = nn.Linear(d_attention, d_chord)
        self.outdur = nn.Linear(d_attention, d_dur)
        self.outpitch = nn.Linear(d_attention, d_pitch)

    def forward_hidden(self, type, barpos, tone, chord, dur, pitch, mask):
        type_emb = self.intype(type)
        barpos_emb = self.inbarpos(barpos)
        tone_emb = self.intone(tone)
        chord_emb = self.inchord(chord)
        dur_emb = self.indur(dur)
        pitch_emb = self.inpitch(pitch)
        
        embs = torch.cat([type_emb, barpos_emb, tone_emb, chord_emb, dur_emb, pitch_emb], dim = -1)
        embs = self.inlinear(embs)
        
        e_outputs = self.encoder(embs, mask)
        pred_type = self.outtype(e_outputs)
        
        return e_outputs, pred_type
    
    def forward_output(self, h, y):
        y_type_emb = self.intype(y[:,:,0:5])
        concat_outputs = torch.cat([h, y_type_emb], dim = -1)
        final_outputs = self.outconcat(concat_outputs)
        
        barpos = self.outbarpos(final_outputs)
        tone = self.outtone(final_outputs)
        chord = self.outchord(final_outputs)
        dur = self.outdur(final_outputs)
        pitch = self.outpitch(final_outputs)
        return barpos, tone, chord, dur, pitch
    
    def forward_sampling(self, h, pred_type):
        type_emb = self.intype(pred_type)
        concat_outputs = torch.cat([h, type_emb], dim = -1)
        final_outputs = self.outconcat(concat_outputs)
        
        barpos = self.outbarpos(final_outputs)
        tone = self.outtone(final_outputs)
        chord = self.outchord(final_outputs)
        dur = self.outdur(final_outputs)
        pitch = self.outpitch(final_outputs)
        return barpos, tone, chord, dur, pitch