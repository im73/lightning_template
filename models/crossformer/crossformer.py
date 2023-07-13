from math import ceil
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import lightning.pytorch as pl
from models.crossformer.cross_encoder import Encoder
from models.crossformer.cross_decoder import Decoder, NewDecoder
from models.crossformer.attn import FullAttention, AttentionLayer, TwoStageAttentionLayer
from models.crossformer.cross_embed import DSW_embedding
from torchaudio import transforms as T
from sklearn.metrics import precision_score, recall_score, f1_score,  accuracy_score


class CrossFormer(pl.LightningModule):
    def __init__(self, model_conf, feature_conf):
        super(CrossFormer, self).__init__()
        self.model_conf = model_conf
        self.data_dim = feature_conf.n_mels if feature_conf.type == "mel" else 1
        self.in_len = int(model_conf.in_len / feature_conf.hop_length) + 1 if feature_conf.type == "mel" \
                                                else int(model_conf.in_len / feature_conf.kernel_size) 
        self.seg_len = model_conf.seg_len
        self.merge_win = model_conf.win_size
        self.d_model = model_conf.d_model
        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * self.in_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * self.in_len / self.seg_len) * self.seg_len
        self.in_len_add = self.pad_in_len - self.in_len
        
        # Embedding
        self.enc_value_embedding = DSW_embedding(self.seg_len, self.d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_in_len // self.seg_len), self.d_model))
        self.pre_norm = nn.LayerNorm(self.d_model)

        # Encoder
        self.encoder = Encoder(model_conf.e_layers, model_conf.win_size, model_conf.d_model, model_conf.n_heads, model_conf.d_ff, block_depth = 1, \
                                    dropout =model_conf.dropout,in_seg_num = (self.pad_in_len // self.seg_len), factor = model_conf.factor)
        
        self.trans_layer = nn.Linear(self.d_model, self.d_model)
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, self.data_dim, (self.pad_out_len // self.seg_len), self.d_model))
        self.decoder = Decoder(self.seg_len, model_conf.e_layers + 1, self.d_model, model_conf.n_heads, model_conf.d_ff, model_conf.dropout, \
                                    out_seg_num = (self.pad_out_len // self.seg_len), factor = model_conf.factor)
        self.out_proj = nn.Sequential(
            nn.Linear(self.data_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_seq):
        batch_size = x_seq.shape[0]
        # x_seq = self.featurizer(x_seq).transpose(2, 1)
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        
        enc_out = self.encoder(x_seq)
        # features = enc_out[-1].reshape(enc_out[-1].shape[0], enc_out[-1].shape[1], -1)
        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)[:, :self.in_len, :]

        # return predict_y[:, :self.out_len, :]
        return self.out_proj(predict_y)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self(x)
        loss = F.binary_cross_entropy(y_hat, y, reduction='none')
        weight_loss = (loss * (y*self.model_conf.weight_loss+(1-y))).mean()
        self.log('train_loss', weight_loss)
        return weight_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self(x)
        
        loss = F.binary_cross_entropy(y_hat, y, reduction='none')
        weight_loss = (loss * (y*self.model_conf.weight_loss+(1-y))).mean()
        preds = (y_hat > self.model_conf.threshold).float().detach().cpu().numpy().astype(int).reshape(-1)
        y = y.detach().cpu().numpy().astype(int).reshape(-1)
        self.log('val_loss', weight_loss)
        metrics = {
            "acc": accuracy_score(preds, y),
            "precison": precision_score(preds, y),
            "recall": recall_score(preds, y),
            "f1": f1_score(preds, y),
            "loss": weight_loss
        }
        self.log_dict(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.float()
        y_hat = self(x)
        
        loss = F.binary_cross_entropy(y_hat, y, reduction='none')
        weight_loss = (loss * (y*self.model_conf.weight_loss+(1-y))).mean()
        preds = (y_hat > self.model_conf.threshold).float().detach().cpu().numpy().astype(int).reshape(-1)
        y = y.detach().cpu().numpy().astype(int).reshape(-1)
        self.log('val_loss', weight_loss)
        metrics = {
            "acc": accuracy_score(preds, y),
            "precison": precision_score(preds, y),
            "recall": recall_score(preds, y),
            "f1": f1_score(preds, y),
            "loss": weight_loss
        }
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.model_conf.optimizer_config.lr)
        return optimizer
    
