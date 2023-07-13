import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from sklearn.metrics import precision_score, recall_score, f1_score,  accuracy_score
from models.FEDformer.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_onlypos
from models.FEDformer.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from models.FEDformer.FourierCorrelation import FourierBlock, FourierCrossAttention
from models.FEDformer.MultiWaveletCorrelation import MultiWaveletCross, MultiWaveletTransform
from models.FEDformer.SelfAttention_Family import FullAttention, ProbAttention
from models.FEDformer.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FEDformer(pl.LightningModule):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, configs):
        super(FEDformer, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model_conf = configs

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_onlypos(configs.enc_in, configs.d_model, dropout=configs.dropout)
        self.dec_embedding = DataEmbedding_onlypos(configs.dec_in, configs.d_model, dropout=configs.dropout)

        if configs.version == 'Wavelets':
            encoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_self_att = MultiWaveletTransform(ich=configs.d_model, L=configs.L, base=configs.base)
            decoder_cross_att = MultiWaveletCross(in_channels=configs.d_model,
                                                  out_channels=configs.d_model,
                                                  seq_len_q=self.seq_len // 2 + self.pred_len,
                                                  seq_len_kv=self.seq_len,
                                                  modes=configs.modes,
                                                  ich=configs.d_model,
                                                  base=configs.base,
                                                  activation=configs.cross_activation)
        else:
            encoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_self_att = FourierBlock(in_channels=configs.d_model,
                                            out_channels=configs.d_model,
                                            seq_len=self.seq_len//2+self.pred_len,
                                            modes=configs.modes,
                                            mode_select_method=configs.mode_select)
            decoder_cross_att = FourierCrossAttention(in_channels=configs.d_model,
                                                      out_channels=configs.d_model,
                                                      seq_len_q=self.seq_len//2+self.pred_len,
                                                      seq_len_kv=self.seq_len,
                                                      modes=configs.modes,
                                                      mode_select_method=configs.mode_select)
        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),

                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=False),
            projection2=nn.Linear(configs.enc_in, configs.c_out, bias=False)
        )

        self.out_activation = nn.Sigmoid()

    def forward(self, x_enc, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_enc.shape[0], self.pred_len, x_enc.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        # trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        # seasonal_init = seasonal_init
        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
    
        
        dec_out = seasonal_part + trend_part

        # if self.output_attention:
        #     return self.out_activation(dec_out), attns
        # else:
    
        return self.out_activation(dec_out)
    
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
        preds = (y_hat >= self.model_conf.threshold).detach().cpu().numpy().astype(int).reshape(-1)
        y = y.detach().cpu().numpy().astype(int).reshape(-1)
        self.log('val_loss', weight_loss)
        print()
        metrics = {
            "acc": sum(preds==y) / len(preds),
            "precison": sum((preds==y) & (y==1)) / sum(preds==1),
            "recall": sum((preds==y) & (y==1)) / sum(y==1),
            "f1": f1_score(preds, y),
            "loss": weight_loss
        }
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.model_conf.optimizer_config.lr)
        return optimizer
        
