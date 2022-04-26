import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from typesql.model.modules.net_utils import run_lstm, col_name_encode, run_attention
import pdb
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# We add an attention based encoder to the model in place of the LSTM.
## Small issue : d_modeland d_hid are same
class AggPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, d_hid=600):
        super(AggPredictor, self).__init__()
        EncoderLayer = nn.TransformerEncoderLayer(d_model=d_hid, nhead=3, dropout=0.0)
        self.pos_encoder = PositionalEncoding(d_hid, dropout=0.0)
        self.encoder = nn.Embedding(N_word, d_hid)
        ## in log first is num_layers
        self.agg_attn = nn.TransformerEncoder(EncoderLayer, num_layers=1)
        '''
        self.agg_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h//2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)

        self.agg_col_name_enc = nn.LSTM(input_size=N_word+N_h,
                hidden_size=N_h//2, num_layers=N_depth,
                batch_first=True, dropout=0.3, bidirectional=True)
        '''
        self.N_h = N_h
        self.agg_att = nn.Linear(N_h, N_h)
        self.sel_att = nn.Linear(d_hid, N_h)
        self.agg_out_se = nn.Linear(N_word, N_h)
        self.agg_out_agg = nn.Linear(N_word, N_h)
        self.agg_out_K = nn.Linear(N_word, N_h)
        self.agg_out_f = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
        self.softmax = nn.Softmax(dim=1) #dim=1

    def forward(self, x_emb_var, x_len, agg_emb_var, col_inp_var=None,
            col_len=None):
        ''' x_emb_var.shape     torch.Size([256, 39, 600])
            x_len.shape         (256,)
            agg_emb_var.shape   torch.Size([256, 6, 600])
            col_inp_var.shape   torch.Size([256, 12, 600])
            col_len.shape       (256,)
        '''
        B = len(x_emb_var)
        max_x_len = max(x_len)

        #h_enc, _ = run_lstm(self.agg_lstm, x_emb_var, x_len)
        #h_enc = run_attention(self.agg_attn, x_emb_var, x_len)
        #breakpoint()
        #x_emb_var = x_emb_var.long() 
        x = x_emb_var
        #x = self.encoder(x_emb_var)
        ## x - > torch.Size([256, 39, 600, 300])
        #* math.sqrt(self.d_model)
        #x = self.pos_encoder(x)
        h_enc = self.agg_attn(x)
        agg_enc = self.agg_out_agg(agg_emb_var)
        ## h_enc -> torch.Size([256, 39, 600])    agg_enc -> torch.Size([256, 6, 120])
        #agg_enc: (B, 6, hid_dim)
        #self.sel_att(h_enc) -> (B, max_x_len, hid_dim) .transpose(1, 2) -> (B, hid_dim, max_x_len)
        #att_val_agg: (B, 6, max_x_len)
        att_val_agg = torch.bmm(agg_enc, self.sel_att(h_enc).transpose(1, 2))

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val_agg[idx, :, num:] = -100

        #att_agg: (B, 6, max_x_len)
        att_agg = self.softmax(att_val_agg.view((-1, max_x_len))).view(B, -1, max_x_len)
        #h_enc.unsqueeze(1) -> (B, 1, max_x_len, hid_dim)
        #att_agg.unsqueeze(3) -> (B, 6, max_x_len, 1)
        #K_agg_expand -> (B, 6, hid_dim)
        K_agg_expand = (h_enc.unsqueeze(1) * att_agg.unsqueeze(3)).sum(2)
        #agg_score = self.agg_out(K_agg)
        ## agg_emb_var, K_agg_expand -> torch.Size([256, 6, 600])     
        agg_score = self.agg_out_f(self.agg_out_se(agg_emb_var) + self.agg_out_K(K_agg_expand)).squeeze()

        return agg_score
