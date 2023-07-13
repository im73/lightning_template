import os
import json
import numpy as np
import pandas as pd
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchaudio import transforms as T
from sklearn.preprocessing import StandardScaler
from util.utils import spectral_residual_transform



class Vibration(Dataset):
    def __init__(self, data_config, feature_config, mode = 'train') -> None:
        super(Vibration, self).__init__()
        L.seed_everything(data_config.seed)
        self.data_root_path = data_config.root_path
        self.label_path = os.path.join(self.data_root_path, data_config.label_path)
        self.label_dict = self._phase_label(self.label_path)
        self.data_path =  os.path.join(self.data_root_path, mode + '.txt')
        self.data_list = self._get_file_path(self.data_path)
        self.feature_config = feature_config.mel_config
        self.scaler = StandardScaler()
        self.featuriszer = T.MelSpectrogram(
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            onesided=True,
            mel_scale="htk",
            **(self.feature_config.__dict__)
        )
        
        
    def _phase_label(self, label_path):
        label_dict = {}
        with open(label_path, "r") as f:
            data = json.load(f)
            # for item in data:
            #     name = item["data"]['timeseries'].split('/')[-1].split('.')[0]
            #     segs = []
            #     for label in item["completions"][0]["result"]:
            #         segs.append([label["value"]["start"], label["value"]["end"]])
            #     label_dict[name] = segs
        return data
    
    def _get_file_path(self, path_file):
        path_list = []
        with open(path_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                path_list.append(os.path.join(self.data_root_path, "files", line.strip()) )
        return path_list
    
    def __getitem__(self, index):
        data_path = self.data_list[index]
        name = data_path.split('/')[-1].split('.')[0]
        label = self.label_dict[name]
        data = self._load_audio(data_path)
        data = self.featuriszer(data).transpose(0, 1)
        label_map = np.zeros((data.shape[0], 1))
        for rg in label:
            left, right = int(rg[0] / self.feature_config.hop_length), int(rg[1] / self.feature_config.hop_length) + 1
            label_map[left:right, :] = 1

        return  data, np.array(label_map)
    
    def _load_audio(self, path, scale=True):
        """
            read csv file
            :param file_path: csv file path
            :return: the first column of csv file
        """
        num = int(path[-1])
        data = pd.read_csv(path[:-2] + ".csv", header=0, usecols=[0], encoding='utf-8')
        data = list(data["timeseries"])
        data = data[num*80000:(num+1)*80000]
        if scale:
            data = self.scaler.fit_transform(np.array([data]))[0]
        return torch.Tensor(data)
    
    def get_original_data(self, index):
        data_path = self.data_list[index]
        data = self._load_audio(data_path, scale=False)
        return data

    def get_original_index(self, index):
        return index * self.feature_config.kernel_size 
    
    def __len__(self):
        return len(self.data_list)
    
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=0)
        
    def forward(self, x):
        # padding on the both ends of time series
        
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class Vibration_mvg(Dataset):
    def __init__(self, data_config, feature_config, mode = 'train') -> None:
        super(Vibration_mvg, self).__init__()
        L.seed_everything(data_config.seed)
        self.data_root_path = data_config.root_path
        self.label_path = os.path.join(self.data_root_path, data_config.label_path)
        self.label_dict = self._phase_label(self.label_path)
        self.data_path =  os.path.join(self.data_root_path, mode + '.txt')
        self.data_list = self._get_file_path(self.data_path)
        self.feature_config = feature_config
        self.featuriszer = moving_avg(kernel_size=feature_config.kernel_size, stride=feature_config.kernel_size)
        self.scaler = StandardScaler()
        
    def _phase_label(self, label_path):
        label_dict = {}
        with open(label_path, "r") as f:
            data = json.load(f)
            # for item in data:
            #     name = item["data"]['timeseries'].split('/')[-1].split('.')[0]
            #     segs = []
            #     for label in item["completions"][0]["result"]:
            #         segs.append([label["value"]["start"], label["value"]["end"]])
            #     label_dict[name] = segs
        return data
    
    def _get_file_path(self, path_file):
        path_list = []
        with open(path_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                path_list.append(os.path.join(self.data_root_path, "files", line.strip()) )
        return path_list
    
    def __getitem__(self, index):
        data_path = self.data_list[index]
        name = data_path.split('/')[-1].split('.')[0]
        label = self.label_dict[name]
        data = self._load_audio(data_path).unsqueeze(0).unsqueeze(-1)
        data = self.featuriszer(data)[0]
        label_map = np.zeros((data.shape[0], 1))
        for rg in label:
            left, right = int(rg[0] / self.feature_config.kernel_size), int(rg[1] / self.feature_config.kernel_size) + 1
            label_map[left:right, :] = 1
        return  data, np.array(label_map)
    
    def _load_audio(self, path, scale=True):
        """
            read csv file
            :param file_path: csv file path
            :return: the first column of csv file
        """
        num = int(path[-1])
        data = pd.read_csv(path[:-2] + ".csv", header=0, usecols=[0], encoding='utf-8')
        data = list(data["timeseries"])
        data = data[num*80000:(num+1)*80000]
        data = spectral_residual_transform(data)
        if scale:
            data = self.scaler.fit_transform(np.array([data]))[0]
        
        return torch.Tensor(data)
    
    def get_original_data(self, index):
        data_path = self.data_list[index]
        data = self._load_audio(data_path, scale=False)
        return data

    def get_original_index(self, index):
        return index * self.feature_config.kernel_size 
    
    def __len__(self):
        return len(self.data_list)

        
class data_processe(object):
    def __init__(self, feature_config):
        self.scaler = StandardScaler()
        self.featuriszer = moving_avg(kernel_size=feature_config.kernel_size, stride=feature_config.kernel_size)

    def load_data(self, csv_path, scale=True):
        data = pd.read_csv(csv_path, header=0, usecols=[0], encoding='utf-8')
        data = list(data["timeseries"])
        data = spectral_residual_transform(data)
        if scale:
            data = self.scaler.fit_transform(np.array([data]))
        else:
            data = np.array([data])
        data = torch.Tensor(data).unsqueeze(-1)
        data = self.featuriszer(data)
        return data
        


        