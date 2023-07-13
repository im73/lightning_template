import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from sklearn.metrics import precision_score, recall_score, f1_score,  accuracy_score
from models.missformer.layers import MissAttentionLayer, EncoderLayer, Encoder, DecoderLayer, Decoder, MissAttention
from models.missformer.embed import DataEmbedding

class MissFormer(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.heads = configs.n_heads
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, split=True)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, split=True)
        self.pred_len = configs.pred_len
        encoder_layer = EncoderLayer(
            MissAttentionLayer(
                MissAttention(output_attention=False), 
                d_model=configs.d_model, 
                n_heads=self.heads, 
                fix_time = False
            ), 
            d_model=configs.d_model, 
            d_ff=configs.d_ff, 
            dropout=configs.dropout, 
            activation="relu", 
            fix_time = False
        )

        self.encoder = Encoder(
            [copy.deepcopy(encoder_layer) for _ in range(configs.e_layers)], 
            norm_layer=nn.LayerNorm(configs.d_model), 
            fix_time = False
        )

        decoder_layer = DecoderLayer(
            MissAttentionLayer(
                MissAttention(output_attention=False), 
                d_model=configs.d_model,
                n_heads=self.heads, 
                fix_time = False
            ), 
            MissAttentionLayer(
                MissAttention(output_attention=False), 
                d_model=configs.d_model,
                n_heads=self.heads, 
                fix_time = False
            ),
            d_model=configs.d_model, 
            d_ff=configs.d_ff, 
            dropout=configs.dropout, 
            activation="relu", 
        )
        self.decoder = Decoder(
            [copy.deepcopy(decoder_layer) for _ in range(configs.d_layers)], 
            norm_layer=nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, x_mask, cross_mask):
        series_avg = x_enc.sum(-1).sum(-1) / x_mask.sum(-1).sum(-1)
        trend = series_avg.unsqueeze(-1).repeat(1, x_enc.shape[1]).unsqueeze(-1)
        seasonal = (x_enc - trend) * x_mask + trend * (1 - x_mask)
        x_dec[:, :self.configs.label_len, :] = seasonal[:, -self.configs.label_len:, :]
        seasonal = seasonal.float()
        x_dec = x_dec.float()
        enc_value_embedding, enc_pos_embedding = self.enc_embedding(seasonal, x_mark_enc)
        enc_val, enc_pos, attns = self.encoder(enc_value_embedding, enc_pos_embedding, attn_mask=x_mask)
        dec_value_embedding, dec_pos_embedding = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_value_embedding, dec_pos_embedding, enc_val, enc_pos, x_mask=cross_mask, cross_mask=x_mask) 
        pred_out = dec_out[:, -self.pred_len:, :] + series_avg.unsqueeze(-1).repeat(1, self.pred_len).unsqueeze(-1).float()
        
        return pred_out, attns
        

    def training_step(self, batch, batch_idx):
        x_enc, x_dec, x_mark_enc, x_mark_dec, mask = batch
        x_enc = x_enc.float()
        x_dec = x_dec.float()
        x_mark_enc = x_mark_enc.float()
        x_mark_dec = x_mark_dec.float() 
        dec_inp = torch.zeros_like(x_dec[:, -self.configs.pred_len:, :]).float()
        dec_inp = torch.cat([x_dec[:, :self.configs.label_len, :], dec_inp], dim=1).float()
        B, _, _ = dec_inp.shape
        cross_mask = torch.ones((B, self.configs.pred_len + self.configs.label_len, 1)).float().to(x_enc.device)
        # cross_mask = torch.cat([mask[:, -self.configs.label_len:, :], cross_mask], dim=1).float()
        
        y_hat, _ = self(x_enc, x_mark_enc, dec_inp, x_mark_dec, mask, cross_mask)
        loss = F.mse_loss(y_hat, x_dec[:, -self.configs.pred_len:, :])
        
        return loss

    def validation_step(self, batch, batch_idx):
        x_enc, x_dec, x_mark_enc, x_mark_dec, mask = batch
        x_enc = x_enc.float()
        x_dec = x_dec.float()
        x_mark_enc = x_mark_enc.float()
        x_mark_dec = x_mark_dec.float() 
        dec_inp = torch.zeros_like(x_dec[:, -self.configs.pred_len:, :]).float()
        dec_inp = torch.cat([x_dec[:, :self.configs.label_len, :], dec_inp], dim=1).float()
        B, _, _ = dec_inp.shape
        cross_mask = torch.ones((B, self.configs.pred_len + self.configs.label_len, 1)).float().to(x_enc.device)
        # cross_mask = torch.cat([mask[:, -self.configs.label_len:, :], cross_mask], dim=1).float()
        y_hat, _ = self(x_enc, x_mark_enc, dec_inp, x_mark_dec, mask, cross_mask)
        val_loss = F.mse_loss(y_hat, x_dec[:, -self.configs.pred_len:, :])
        mae = F.l1_loss(y_hat, x_dec[:, -self.configs.pred_len:, :])
        metrics = {"mae": mae, "mse": val_loss}
        self.log_dict(metrics)
        return metrics
    
    def test_step(self, batch, batch_idx):
        x_enc, x_dec, x_mark_enc, x_mark_dec, mask = batch
        x_enc = x_enc.float()
        x_dec = x_dec.float()
        x_mark_enc = x_mark_enc.float()
        x_mark_dec = x_mark_dec.float() 
        dec_inp = torch.zeros_like(x_dec[:, -self.configs.pred_len:, :]).float()
        dec_inp = torch.cat([x_dec[:, :self.configs.label_len, :], dec_inp], dim=1).float()
        B, _, _ = dec_inp.shape
        cross_mask = torch.ones((B, self.configs.pred_len + self.configs.label_len, 1)).float().to(x_enc.device)
        # cross_mask = torch.cat([mask[:, -self.configs.label_len:, :], cross_mask], dim=1).float()
        y_hat, _ = self(x_enc, x_mark_enc, dec_inp, x_mark_dec, mask, cross_mask)
        val_loss = F.mse_loss(y_hat, x_dec[:, -self.configs.pred_len:, :])
        mae = F.l1_loss(y_hat, x_dec[:, -self.configs.pred_len:, :])
        metrics = {"mae": mae, "mse": val_loss}
        self.log_dict(metrics)
        return metrics

    def predict_step(self, batch, batch_idx):
        x_enc, x_dec, x_mark_enc, x_mark_dec, mask = batch
        x_enc = x_enc.float()
        x_dec = x_dec.float()
        x_mark_enc = x_mark_enc.float()
        x_mark_dec = x_mark_dec.float() 
        dec_inp = torch.zeros_like(x_dec[:, -self.configs.pred_len:, :]).float()
        dec_inp = torch.cat([x_dec[:, :self.configs.label_len, :], dec_inp], dim=1).float()
        B, _, _ = dec_inp.shape
        cross_mask = torch.ones((B, self.configs.pred_len + self.configs.label_len, 1)).float().to(x_enc.device)
        y_hat, _ = self(x_enc, x_mark_enc, dec_inp, x_mark_dec, mask, cross_mask)
        return x_enc,x_dec[:, -self.configs.pred_len:, :], y_hat, mask
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.configs.lr)
        return optimizer
    

