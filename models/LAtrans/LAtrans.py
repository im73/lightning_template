import os
import math
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as pl
from sklearn.metrics import precision_score, recall_score, f1_score,  accuracy_score
from models.LAtrans.Embed import DataEmbedding, DataEmbedding_onlypos
from models.LAtrans.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from models.LAtrans.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp


class LAtrans(pl.LightningModule):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """
    def __init__(self, configs):
        super(LAtrans, self).__init__()
        self.configs = configs
        self.seq_len = configs.window_size
        self.output_attention = False
        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        self.enc_embedding = DataEmbedding_onlypos(configs.dim, configs.d_model)
        self.dec_embedding = DataEmbedding_onlypos(configs.dim, configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, output_attention=True),
                        configs.d_model, configs.n_heads, configs.min_len),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    activation=configs.activation
                ) for l in range(configs.layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.factor, output_attention=True),
                        configs.d_model, configs.n_heads, configs.min_len),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, output_attention=True),
                        configs.d_model, configs.n_heads, configs.min_len),
                    configs.d_model,
                    configs.dim,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    activation=configs.activation,
                )
                for l in range(1)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.dim, bias=True)
        )

        # self.weight_para = nn.Linear(configs.dim, 1)
        self.out_projection = nn.Linear(configs.d_model, configs.dim, bias=True)
    def forward(self, x_enc):
        # decomp init
        # seasonal_init, trend_init = self.decomp(x_enc)
        # enc
        enc_out = self.enc_embedding(x_enc)
        enc_out, attns, delay = self.encoder(enc_out)
        # dec
        # dec_out = self.dec_embedding(seasonal_init)
        
        # seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
        #                                          trend=trend_init)
        # final
        prediction = self.out_projection(enc_out)
        # pre_prediction = seasonal_part + trend_part
        return prediction, attns, delay
        # return prediction
    
    def training_step(self, batch, batch_idx):
        x, _ = batch

        y_hat = self(x)[0]
        anoscore = F.mse_loss(y_hat, x)
        self.log('train_loss', anoscore)
        return anoscore
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch

        y_hat = self(x)[0]
        anoscore = F.mse_loss(y_hat, x)
        self.log('val_loss', anoscore)
        metrics = {
            "val_loss": anoscore,
        }
        self.log_dict(metrics)
        return metrics
    
    def predict_step(self, batch, batch_idx):
        # enable Monte Carlo Dropout
        x, y = batch
        y_hat = self(x)[0]
        anoscore = F.mse_loss(y_hat, x, reduction='none')
        # take average of `self.mc_iteration` iterations
        return anoscore, y
    
    def predict_sample(self, batch):
        # enable Monte Carlo Dropout
        x = batch

        y_hat = self(x)[0]
        anoscore = F.mse_loss(y_hat, x, reduction='none')
        # take average of `self.mc_iteration` iterations
        return anoscore, x
    
    def cal_metric(self, score, y):
        
        score = (score - np.min(score)) / (np.max(score) - np.min(score))
        max_f1 = 0
        max_recall = 0
        max_precision = 0
        best_threshold = 0
        for threshold in range(1, 100):
            threshold = 0.00000001 * threshold
            pred_point = (score > threshold).astype(float)
            if f1_score(y, pred_point) > max_f1:
                best_threshold = threshold
                max_f1 = f1_score(y, pred_point)
                max_recall = recall_score(y, pred_point)
                max_precision = precision_score(y, pred_point)
            print(f1_score(y, pred_point), recall_score(y, pred_point), precision_score(y, pred_point), best_threshold)
        return max_f1, max_recall, max_precision, best_threshold


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.configs.optimizer_config.lr)
        return optimizer
    