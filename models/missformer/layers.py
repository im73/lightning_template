import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
class MissAttention(nn.Module):
    def __init__(self, factor=5, scale=None, attention_dropout=0.1, output_attention=False, split=False):
        super(MissAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout_v = nn.Dropout(attention_dropout)
        self.dropout_p = nn.Dropout(attention_dropout)
        
    def forward(self, q_val, q_pos, k_val, k_pos, v_val, v_pos, attn_mask):
        
        B, L, H, E = q_pos.shape
        _, S, _, D = k_pos.shape
        scale = self.scale or 1./sqrt(E)

        attn_mask = attn_mask.bool()
        attn_mask = attn_mask.unsqueeze(1).repeat(1, H, 1, 1)

        scores_vv = torch.einsum("blhe,bshe->bhls", q_val, k_val)
        scores_pp = torch.einsum("blhe,bshe->bhls", q_pos, k_pos)
        # scores_pv = torch.einsum("blhe,bshe->bhls", q_pos, k_val)
        # scores_vp = torch.einsum("blhe,bshe->bhls", q_val, k_pos)
        # import pdb; pdb.set_trace()
        
        # scores = (scores_vv+scores_pp).masked_fill_(~attn_mask, -torch.inf)
        # A_v = self.dropout_v(torch.softmax(scale * scores, dim=-1))
        # A_p = self.dropout_p(torch.softmax(scale * scores_pp, dim=-1))
        # A = A_v.masked_fill_(~attn_mask, 0) + A_p
        # V = torch.einsum("bhls,bshe->blhe", A, v_val)
        # v_pos = torch.einsum("bhls,bshe->blhe", scores_pp, v_pos)

        scores = (scores_vv+scores_pp)
        # scores = torch.einsum("blhe,bshe->bhls", q_val+q_pos, k_val+k_pos)
        A = self.dropout_v(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshe->blhe", A, v_val)
        v_pos = torch.einsum("bhls,bshe->blhe", A, v_pos)

        if self.output_attention:
            return (V, v_pos, A)
        else:
            return (V, v_pos,  None)

class MissAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, fix_time = False):
        super(MissAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection_v = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection_v = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection_v = nn.Linear(d_model, d_values * n_heads)
        
        self.query_projection_p = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection_p = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection_p = nn.Linear(d_model, d_values * n_heads)

        self.out_projection_v = nn.Linear(d_values * n_heads, d_model)
        self.out_projection_p = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.fix_time = fix_time
        
    def forward(self, q_val, q_pos, k_val, k_pos, v_val, v_pos, attn_mask=None):
        B, L, _ = q_pos.shape
        _, S, _ = k_pos.shape
        H = self.n_heads
        ori_time = v_pos
        q_pos = self.query_projection_p(q_pos).view(B, L, H, -1)
        q_val = self.query_projection_v(q_val).view(B, L, H, -1)
        k_pos = self.key_projection_p(k_pos).view(B, S, H, -1)
        k_val = self.key_projection_v(k_val).view(B, S, H, -1)
        v_pos = self.value_projection_p(v_pos).view(B, S, H, -1)
        v_val = self.value_projection_v(v_val).view(B, S, H, -1)
        
        out_v, out_p, attn = self.inner_attention(
            q_val, 
            q_pos, 
            k_val, 
            k_pos, 
            v_val, 
            v_pos,
            attn_mask
        )

        
        out_v = out_v.reshape(B, L, -1)
        out_p = out_p.reshape(B, L, -1)
        if not self.fix_time:
            return self.out_projection_v(out_v),self.out_projection_p(out_p), attn
        else:
            return self.out_projection_v(out_v),ori_time, attn


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", fix_time = False):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_v, x_pos, attn_mask=None):
        attn_mask = torch.einsum("bld,bsd->bls", attn_mask, attn_mask)
        new_x, new_pos, attn = self.attention(
            x_v, x_pos, x_v, x_pos, x_v, x_pos,
            attn_mask=attn_mask
        )
        new_x = new_x + self.dropout(new_x)

        # y = x = self.norm1(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(new_x), new_pos, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None, fix_time = False):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x_v, x_pos, attn_mask=None):
        # x [B, L, D]
        attns = []
        # if self.conv_layers is not None:
        #     for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
        #         x, attn = attn_layer(x, attn_mask=attn_mask)
        #         x = conv_layer(x)
        #         attns.append(attn)
        #     x, attn = self.attn_layers[-1](x)
        #     attns.append(attn)
        # else:
        for attn_layer in self.attn_layers:
            x_v, x_pos, attn = attn_layer(x_v, x_pos, attn_mask=attn_mask)
        
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x_v)
        return x, x_pos, attns

class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x_val, x_pos,  cross_val, cross_pos, x_mask=None, cross_mask=None):
        attn_mask = torch.einsum("bld,bsd->bls", x_mask, x_mask)
        v, x_pos, attns = self.self_attention(
            x_val, x_pos, x_val, x_pos, x_val, x_pos,
            attn_mask=attn_mask
        )
        x_val = x_val + self.dropout(v)
        x_val = self.norm1(x_val)
        attn_mask = torch.einsum("bld,bsd->bls", x_mask, cross_mask)
        v, x_pos, attns = self.cross_attention(
            x_val, x_pos, cross_val, cross_pos,  cross_val, cross_pos,
            attn_mask=attn_mask
        )
        x_val = x_val + self.dropout(v)
        # y = x = self.norm2(x)
        # y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        # y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x_val), x_pos


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x_val, x_pos, cross_val, cross_pos, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x_val, x_pos = layer(x_val, x_pos, cross_val, cross_pos, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x_val = self.norm(x_val)

        if self.projection is not None:
            x_val = self.projection(x_val)
        return x_val
if __name__ == '__main__':
    B = 2
    L = 5
    D = 512
    H = 8
    x = torch.randn(B, L, D)
    x_pos = torch.randn(B, L, D)
    attn_mask = torch.ones(B, L, 1)
    attn_mask[:, L//2, :] = 0
    encoder_layer = DecoderLayer(
        MissAttentionLayer(
           MissAttention(output_attention=False), 
            d_model=D, 
            n_heads=H, 
            fix_time = False
        ), 
        MissAttentionLayer(
           MissAttention(output_attention=False), 
            d_model=D, 
            n_heads=H, 
            fix_time = False
        ),
        d_model=D, 
        d_ff=4*D, 
        dropout=0.1, 
        activation="relu", 
    )
    decoder = Decoder(
        [copy.deepcopy(encoder_layer) for _ in range(1)], 
        norm_layer=nn.LayerNorm(D), 
    )
    cross = torch.randn(B, L*2, D)
    cross_pos = torch.randn(B, L*2, D)
    cross_mask = torch.ones(B, L*2, 1)
    cross_mask[:, L:, :] = 0
    y = decoder(x, x_pos, cross, cross_pos, x_mask=attn_mask, cross_mask=cross_mask) 
    
