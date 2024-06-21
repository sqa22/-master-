import torch
from torch import nn
import math
from torch.nn import LayerNorm
from torch.nn import Dropout
from base_model import SequenceModel


import torch
from torch import nn
import math
from base_model import SequenceModel

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:x.shape[1], :]

class SAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.temperature = math.sqrt(self.d_model/nhead)

        self.qtrans = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.ktrans = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.vtrans = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        attn_dropout_layer = [Dropout(p=dropout) for _ in range(nhead)]
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.ReLU(),
            Dropout(p=dropout),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x.permute(0, 2, 1)  # Change to [N, D, T]
        q = self.qtrans(x).permute(0, 2, 1)  # Back to [N, T, D]
        k = self.ktrans(x).permute(0, 2, 1)
        v = self.vtrans(x).permute(0, 2, 1)

        dim = int(self.d_model/self.nhead)
        att_output = []
        for i in range(self.nhead):
            qh = q[:, :, i * dim:(i + 1) * dim]
            kh = k[:, :, i * dim:(i + 1) * dim]
            vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.cat(att_output, dim=2)

        xt = x.permute(0, 2, 1) + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt.permute(0, 2, 1)).permute(0, 2, 1)

        return att_output

class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.qtrans = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.ktrans = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)
        self.vtrans = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

        self.attn_dropout = [Dropout(p=dropout) for _ in range(nhead)]
        self.attn_dropout = nn.ModuleList(self.attn_dropout)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1),
            nn.ReLU(),
            Dropout(p=dropout),
            nn.Conv1d(d_model, d_model, kernel_size=1),
            Dropout(p=dropout)
        )

    def forward(self, x):
        x = self.norm1(x)
        x = x.permute(0, 2, 1)  # Change to [N, D, T]
        q = self.qtrans(x).permute(0, 2, 1)  # Back to [N, T, D]
        k = self.ktrans(x).permute(0, 2, 1)
        v = self.vtrans(x).permute(0, 2, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            qh = q[:, :, i * dim:(i + 1) * dim]
            kh = k[:, :, i * dim:(i + 1) * dim]
            vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.cat(att_output, dim=2)

        xt = x.permute(0, 2, 1) + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt.permute(0, 2, 1)).permute(0, 2, 1)

        return att_output

class Gate(nn.Module):
    def __init__(self, d_input, d_output, beta=1.0):
        super().__init__()
        self.trans = nn.Linear(d_input, d_output)
        self.d_output = d_output
        self.t = beta

    def forward(self, gate_input):
        output = self.trans(gate_input)
        output = torch.softmax(output / self.t, dim=-1)
        return self.d_output * output

class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=1, bias=False)

    def forward(self, z):
        z = z.permute(0, 2, 1)
        h = self.conv(z).permute(0, 2, 1)
        query = h[:, -1, :].unsqueeze(-1)
        lam = torch.matmul(h, query).squeeze(-1)
        lam = torch.softmax(lam, dim=1).unsqueeze(1)
        output = torch.matmul(lam, z.permute(0, 2, 1)).squeeze(1)
        return output

class MASTER(nn.Module):
    def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221, beta=None):
        super(MASTER, self).__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.conv = nn.Conv1d(in_channels=d_feat, out_channels=d_model, kernel_size=1)
        self.pos_encoding = PositionalEncoding(d_model)
        self.intra_stock_att = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        self.inter_stock_att = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        self.temporal_att = TemporalAttention(d_model=d_model)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index]
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        src = src.permute(0, 2, 1)
        src = self.conv(src)
        src = src.permute(0, 2, 1)

        src = self.pos_encoding(src)
        src = self.intra_stock_att(src)
        src = self.inter_stock_att(src)
        src = self.temporal_att(src)
        output = self.decoder(src).squeeze(-1)

        return output

class MASTERModel(SequenceModel):
    def __init__(self, d_feat: int = 20, d_model: int = 64, t_nhead: int = 4, s_nhead: int = 2, gate_input_start_index=None, gate_input_end_index=None,
                 T_dropout_rate=0.5, S_dropout_rate=0.5, beta=5.0, **kwargs):
        super(MASTERModel, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_feat = d_feat

        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index

        self.T_dropout_rate = T_dropout_rate
        self.S_dropout_rate = S_dropout_rate
        self.t_nhead = t_nhead
        self.s_nhead = s_nhead
        self.beta = beta

        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                            T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                            gate_input_start_index=self.gate_input_start_index,
                            gate_input_end_index=self.gate_input_end_index, beta=self.beta)
        super(MASTERModel, self).init_model()
