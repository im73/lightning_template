import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler
from pytorch_lightning import seed_everything
from util.utils import StandardScaler, load_yml
from util.time_features import time_features
import copy 
import warnings
warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, data_config, mode = 'train'):
        # size [seq_len, label_len, pred_len]
        # info
        self.config = data_config
        self.target = self.config.target
        self.seq_len = self.config.seq_len
        self.label_len = self.config.label_len
        self.pred_len = self.config.pred_len
        # init
        assert mode in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[mode]
        
        self.features = self.config.features
        self.scale = self.config.scale
        self.freq = self.config.freq
        
        self.data_path = self.config.data_path
        self.mask_percentage = self.config.mask_percentage
        print(" mask_paercentage : {}".format(self.mask_percentage))
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.data_path)

        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=0, freq=self.freq)

        self.data_x = data[border1:border2]
        
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        self.data_x = np.array(self.data_x)
        self.data_y = copy.deepcopy(np.array(self.data_y))
        self.mask_x = np.ones((self.data_x.shape[0], 1))

        seed_everything(42)
        if self.mask_percentage !=0 :
            nan_idx = np.random.choice(len(self.data_x), size = int(len(self.data_x) * self.mask_percentage), replace=False)
            self.data_x[nan_idx, :] = 0
            self.mask_x[nan_idx, :] = 0
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        pre_train_mask = np.zeros_like(seq_x)
        
        seq_y = self.data_y[r_begin:r_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, self.mask_x[s_begin:s_end]
    
    def __len__(self):
        return (len(self.data_x) - self.seq_len- self.pred_len + 1) 

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def get_original_series(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        return self.data_y[s_begin:s_end]

# class Dataset_ETT_minute(Dataset):
#     def __init__(self, root_path, flag='train', size=None, 
#                  features='S', data_path='ETTm1.csv', 
#                  target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24*4*4
#             self.label_len = 24*4
#             self.pred_len = 24*4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train':0, 'val':1, 'test':2}
#         self.set_type = type_map[flag]
        
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.freq = freq
        
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))

#         border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
#         border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
        
#         if self.features=='M' or self.features=='MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features=='S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
            
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
#         self.data_x = data[border1:border2]
#         self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
    
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         if self.inverse:
#             seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
#         else:
#             seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark
    
#     def __len__(self):
#         return len(self.data_x) - self.seq_len - self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)


# class Dataset_Custom(Dataset):
#     def __init__(self, root_path, flag='train', size=None, 
#                  features='S', data_path='ETTh1.csv', 
#                  target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24*4*4
#             self.label_len = 24*4
#             self.pred_len = 24*4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['train', 'test', 'val']
#         type_map = {'train':0, 'val':1, 'test':2}
#         self.set_type = type_map[flag]
        
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cols=cols
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         # cols = list(df_raw.columns); 
#         if self.cols:
#             cols=self.cols.copy()
#             cols.remove(self.target)
#         else:
#             cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
#         df_raw = df_raw[['date']+cols+[self.target]]

#         num_train = int(len(df_raw)*0.7)
#         num_test = int(len(df_raw)*0.2)
#         num_vali = len(df_raw) - num_train - num_test
#         border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
#         border2s = [num_train, num_train+num_vali, len(df_raw)]
#         border1 = border1s[self.set_type]
#         border2 = border2s[self.set_type]
        
#         if self.features=='M' or self.features=='MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features=='S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             train_data = df_data[border1s[0]:border2s[0]]
#             self.scaler.fit(train_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
            
#         df_stamp = df_raw[['date']][border1:border2]
#         df_stamp['date'] = pd.to_datetime(df_stamp.date)
#         data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

#         self.data_x = data[border1:border2]
#         if self.inverse:
#             self.data_y = df_data.values[border1:border2]
#         else:
#             self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
    
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len 
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         if self.inverse:
#             seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
#         else:
#             seq_y = self.data_y[r_begin:r_end]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark
    
#     def __len__(self):
#         return len(self.data_x) - self.seq_len- self.pred_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

# class Dataset_Pred(Dataset):
#     def __init__(self, root_path, flag='pred', size=None, 
#                  features='S', data_path='ETTh1.csv', 
#                  target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
#         # size [seq_len, label_len, pred_len]
#         # info
#         if size == None:
#             self.seq_len = 24*4*4
#             self.label_len = 24*4
#             self.pred_len = 24*4
#         else:
#             self.seq_len = size[0]
#             self.label_len = size[1]
#             self.pred_len = size[2]
#         # init
#         assert flag in ['pred']
        
#         self.features = features
#         self.target = target
#         self.scale = scale
#         self.inverse = inverse
#         self.timeenc = timeenc
#         self.freq = freq
#         self.cols=cols
#         self.root_path = root_path
#         self.data_path = data_path
#         self.__read_data__()

#     def __read_data__(self):
#         self.scaler = StandardScaler()
#         df_raw = pd.read_csv(os.path.join(self.root_path,
#                                           self.data_path))
#         '''
#         df_raw.columns: ['date', ...(other features), target feature]
#         '''
#         if self.cols:
#             cols=self.cols.copy()
#             cols.remove(self.target)
#         else:
#             cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
#         df_raw = df_raw[['date']+cols+[self.target]]
        
#         border1 = len(df_raw)-self.seq_len
#         border2 = len(df_raw)
        
#         if self.features=='M' or self.features=='MS':
#             cols_data = df_raw.columns[1:]
#             df_data = df_raw[cols_data]
#         elif self.features=='S':
#             df_data = df_raw[[self.target]]

#         if self.scale:
#             self.scaler.fit(df_data.values)
#             data = self.scaler.transform(df_data.values)
#         else:
#             data = df_data.values
            
#         tmp_stamp = df_raw[['date']][border1:border2]
#         tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
#         pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
#         df_stamp = pd.DataFrame(columns = ['date'])
#         df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
#         data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

#         self.data_x = data[border1:border2]
#         if self.inverse:
#             self.data_y = df_data.values[border1:border2]
#         else:
#             self.data_y = data[border1:border2]
#         self.data_stamp = data_stamp
    
#     def __getitem__(self, index):
#         s_begin = index
#         s_end = s_begin + self.seq_len
#         r_begin = s_end - self.label_len
#         r_end = r_begin + self.label_len + self.pred_len

#         seq_x = self.data_x[s_begin:s_end]
#         if self.inverse:
#             seq_y = self.data_x[r_begin:r_begin+self.label_len]
#         else:
#             seq_y = self.data_y[r_begin:r_begin+self.label_len]
#         seq_x_mark = self.data_stamp[s_begin:s_end]
#         seq_y_mark = self.data_stamp[r_begin:r_end]

#         return seq_x, seq_y, seq_x_mark, seq_y_mark
    
#     def __len__(self):
#         return len(self.data_x) - self.seq_len + 1

#     def inverse_transform(self, data):
#         return self.scaler.inverse_transform(data)

if __name__ == '__main__':
    yml_path = './config/missformer.yml'
    config = load_config(yml_path)
    dataset = Dataset_ETT_hour(config.data_config)
    