import torch
from torch import nn
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
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
        self.temperature = math.sqrt(self.d_model / nhead)

        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        attn_dropout_layer = []
        for i in range(nhead):
            attn_dropout_layer.append(Dropout(p=dropout))
        self.attn_dropout = nn.ModuleList(attn_dropout_layer)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        # x = self.norm1(x)
        q = self.qtrans(x).transpose(0, 1)
        k = self.ktrans(x).transpose(0, 1)
        v = self.vtrans(x).transpose(0, 1)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]

            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)) / self.temperature, dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh).transpose(0, 1))
        att_output = torch.cat(att_output, dim=-1)
        
        x = self.norm1(x)
        
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

        return att_output

class TAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.qtrans = nn.Linear(d_model, d_model, bias=False)
        self.ktrans = nn.Linear(d_model, d_model, bias=False)
        self.vtrans = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = []
        if dropout > 0:
            for i in range(nhead):
                self.attn_dropout.append(Dropout(p=dropout))
            self.attn_dropout = nn.ModuleList(self.attn_dropout)

        self.norm1 = LayerNorm(d_model, eps=1e-5)
        self.norm2 = LayerNorm(d_model, eps=1e-5)
        self.ffn = nn.Sequential(
            Linear(d_model, d_model),
            nn.ReLU(),
            Dropout(p=dropout),
            Linear(d_model, d_model),
            Dropout(p=dropout)
        )

    def forward(self, x):
        # x = self.norm1(x)
        q = self.qtrans(x)
        k = self.ktrans(x)
        v = self.vtrans(x)

        dim = int(self.d_model / self.nhead)
        att_output = []
        for i in range(self.nhead):
            if i == self.nhead - 1:
                qh = q[:, :, i * dim:]
                kh = k[:, :, i * dim:]
                vh = v[:, :, i * dim:]
            else:
                qh = q[:, :, i * dim:(i + 1) * dim]
                kh = k[:, :, i * dim:(i + 1) * dim]
                vh = v[:, :, i * dim:(i + 1) * dim]
            atten_ave_matrixh = torch.softmax(torch.matmul(qh, kh.transpose(1, 2)), dim=-1)
            if self.attn_dropout:
                atten_ave_matrixh = self.attn_dropout[i](atten_ave_matrixh)
            att_output.append(torch.matmul(atten_ave_matrixh, vh))
        att_output = torch.cat(att_output, dim=-1)
        
        
        x = self.norm1(x)
        
        xt = x + att_output
        xt = self.norm2(xt)
        att_output = xt + self.ffn(xt)

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

# class TemporalAttention(nn.Module):
#     def __init__(self, d_model):
#         super().__init__()
#         self.trans = nn.Linear(d_model, d_model, bias=False)

#     def forward(self, z):
#         h = self.trans(z)
#         query = h[:, -1, :].unsqueeze(-1)
#         lam = torch.matmul(h, query).squeeze(-1)
#         lam = torch.softmax(lam, dim=1).unsqueeze(1)
#         output = torch.matmul(lam, z).squeeze(1)
#         return output


class TemporalAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.trans = nn.Linear(d_model, d_model, bias=False)

    def forward(self, z):
        # h = self.trans(z) # [N, T, D]
        # query = h[:, -1, :].unsqueeze(-1)
        # lam = torch.matmul(h, query).squeeze(-1)  # [N, T, D] --> [N, T]
        # lam = torch.softmax(lam, dim=1).unsqueeze(1)
        # output = torch.matmul(lam, z).squeeze(1)  # [N, 1, T], [N, T, D] --> [N, 1, D]
        output = (z[:, -1:, :]).squeeze(1)
        # print(z.shape)   
        # print(output.shape)
        return output


# class MultiScaleFeatureAggregation(nn.Module):
#     def __init__(self, d_model, scales):
#         super(MultiScaleFeatureAggregation, self).__init__()
#         self.scales = scales
#         self.convs = nn.ModuleList([
#             nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=scale, padding=scale//2)
#             for scale in scales
#         ])
#         self.norm = LayerNorm(d_model)

    # def forward(self, x):
    #     # x shape: (N, T, D)
    #     x = x.transpose(1, 2)  # (N, D, T)
    #     features = [conv(x) for conv in self.convs]
    #     features = torch.cat(features, dim=2)  # (N, D, T * len(scales))
    #     features = features.transpose(1, 2)  # (N, T * len(scales), D)
    #     return self.norm(features)

class MASTER(nn.Module):
    def __init__(self, d_feat=158, d_model=256, t_nhead=4, s_nhead=2, T_dropout_rate=0.5, S_dropout_rate=0.5,
                 gate_input_start_index=158, gate_input_end_index=221, beta=None, scales=[3, 5, 7]):
        super(MASTER, self).__init__()
        self.gate_input_start_index = gate_input_start_index
        self.gate_input_end_index = gate_input_end_index
        self.d_gate_input = (gate_input_end_index - gate_input_start_index)
        self.feature_gate = Gate(self.d_gate_input, d_feat, beta=beta)

        self.feature_layer = nn.Linear(d_feat, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.intra_stock_attention = TAttention(d_model=d_model, nhead=t_nhead, dropout=T_dropout_rate)
        # self.multi_scale_aggregation = MultiScaleFeatureAggregation(d_model=d_model, scales=scales)
        self.inter_stock_attention = SAttention(d_model=d_model, nhead=s_nhead, dropout=S_dropout_rate)
        self.temporal_attention = TemporalAttention(d_model=d_model)
        self.output_layer = nn.Linear(d_model, 1)

    def forward(self, x):
        src = x[:, :, :self.gate_input_start_index]
        gate_input = x[:, -1, self.gate_input_start_index:self.gate_input_end_index]
        src = src * torch.unsqueeze(self.feature_gate(gate_input), dim=1)

        x = self.feature_layer(src)
        x = self.positional_encoding(x)
        x = self.intra_stock_attention(x)
        # x = self.multi_scale_aggregation(x)
        x = self.inter_stock_attention(x)
        x = self.temporal_attention(x)

        output = self.output_layer(x).squeeze(-1)
        return output

class MASTERModel(SequenceModel):
    def __init__(self, d_feat: int = 20, d_model: int = 64, t_nhead: int = 4, s_nhead: int = 2, gate_input_start_index=None, gate_input_end_index=None,
                 T_dropout_rate=0.5, S_dropout_rate=0.5, beta=5.0, scales=[3, 5, 7], **kwargs):
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
        self.scales = scales
        self.init_model()

    def init_model(self):
        self.model = MASTER(d_feat=self.d_feat, d_model=self.d_model, t_nhead=self.t_nhead, s_nhead=self.s_nhead,
                            T_dropout_rate=self.T_dropout_rate, S_dropout_rate=self.S_dropout_rate,
                            gate_input_start_index=self.gate_input_start_index,
                            gate_input_end_index=self.gate_input_end_index, beta=self.beta, scales=self.scales)
        super(MASTERModel, self).init_model()
