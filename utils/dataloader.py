import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random 
import os 
import torch
import gc
import math
import json

from tqdm import tqdm
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, WeightedRandomSampler
from imblearn.over_sampling import RandomOverSampler

class TS_EHR_Dataset(Dataset):
    def __init__(self, df, args, data_type):

        super(TS_EHR_Dataset, self).__init__()
        
        self.df = df
        self.args = args
        self.data_type = data_type
        
        # label_cols = ['mortality_inhospital', 'mortality_inunit', 'los_3', 'los_7', 'CF_Annotation', 'CF_next_8h', 'AKI_Annotation', 'AKI_next_8h']

        # self.X_cols = sorted(list(set(self.df.columns) - 
        #                           set(['subject_id', 'hadm_id', 'stay_id','Frequency_Spectrum', 'Spectrum_group'] + label_cols)))

        label_cols = ['mortality_inhospital', 'los_3', 'los_7', 'CF_Annotation', 'CF_next_8h', 'AKI_Annotation', 'AKI_next_8h']
        self.X_cols = sorted(list(set(self.df.columns) - set(['stay_id', 'time_since_ICU'] + label_cols)))
        
        if args.use_pad:
            X_data, self.X_mask = self.reshape_data_matrix()
        else:
            X_data = self.reshape_data_matrix()
            
        y_data = self.df[self.df['time_since_ICU']== args.seq_len - 1].groupby('stay_id')[self.args.label].mean().values
        
        ## Random Oversampling
        # if self.args.oversampling and self.data_type == 'train':
        #     sampler = RandomOverSampler(sampling_strategy='auto', random_state = args.seed)
        #     sample_idx = np.arange(X_data.shape[0]).reshape(-1,1) 
        #     idx_res, y_res = sampler.fit_resample(sample_idx, y_data)
        #     X_res = X_data[idx_res.ravel()]

        #     X_data, y_data = X_res, y_res
            
        self.X_data = torch.from_numpy(X_data).float() # (N, T, C)
        self.y_data = torch.from_numpy(y_data).long()  # (N)

        print(f'Input X, Y Shape : {self.X_data.shape}, {self.y_data.shape}')
        print(f'Unique values of the label: {np.unique(y_data)}')
        
        # For univariate time series
        if len(self.X_data.shape) < 3:
            self.X_data = self.X_data.unsqueeze(2)

        self.len = self.X_data.shape[0]

    def reshape_data_matrix(self):

        df = self.df.sort_values(['stay_id', 'time_since_ICU'])
        
        if self.args.use_pad:
            result = []
            result_mask = [] # Distinguish to pad or observed.
            for stay in tqdm(df.stay_id.unique()):
                ts = df[df['stay_id'] == stay][self.X_cols].values
                pad_ts, mask_ts = self._sliding_window_with_padding(ts, self.args.seq_len)
                result.append(pad_ts) # For various time sequence length.
                result_mask.append(mask_ts) # (N, time_seq)
                
            result = np.concatenate(result, axis = 0)
            result_mask = np.concatenate(result_mask, axis = 0)
            
            return result, result_mask
        
        else:
            grouped_df = df.groupby(['stay_id','time_since_ICU'])[self.X_cols].mean() # Use raw value at each time seqeunce.
            result = grouped_df.values.reshape(self.df['stay_id'].nunique(), self.args.seq_len, len(self.X_cols))
                
            return result
    
    def _sliding_window_with_padding(self, arr, window_size=25):
        n_time, _ = arr.shape
        n_windows = int(np.ceil(n_time / window_size))
        windows = []
        masks = []
        
        for i in range(n_windows):
            start = i * window_size
            end = start + window_size
            window = arr[start:end]
            original_length = window.shape[0]
            # Mask matrix for original data. 
            mask = np.ones(original_length, dtype=bool)
            
            # 만약 window 길이가 window_size보다 작으면 nan으로 패딩하고, 마스크도 False로 패딩
            if original_length < window_size:
                pad_length = window_size - original_length
                window = np.pad(window, ((0, pad_length), (0, 0)), 
                                mode='constant', constant_values=np.nan)
                mask = np.pad(mask, (0, pad_length), mode='constant', constant_values=False)
                
            windows.append(window)
            masks.append(mask)
    
        return np.array(windows), np.array(masks)

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def dl_data_generator(data_dict, args):

    # Loading datasets
    if args.train_type == 'general': 
        if args.mode == 'train':
            train_dataset = TS_EHR_Dataset(data_dict['train'], args, 'train')
            valid_dataset = TS_EHR_Dataset(data_dict['valid'], args, 'valid')
            test_dataset = TS_EHR_Dataset(data_dict['test'], args, 'test')

            # Dataloaders
            batch_size = args.batch_size

            # WeightedRandomSampler
            if args.oversampling:
                class_sample_counts = np.bincount(train_dataset.y_data)
                class_weights = 1.0 / class_sample_counts
                print('Class Weights',class_sample_counts, class_weights)
                sample_weights = class_weights[train_dataset.y_data]
                
                sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),  
                                                num_samples=len(sample_weights),             
                                                replacement=True)
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                            sampler=sampler, num_workers=0)
            else:
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                            shuffle=False, num_workers=0)

            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                                    shuffle=False, num_workers=0)
            
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                    shuffle=False, num_workers=0)
            
            return train_loader, valid_loader, test_loader
        
        else:
            for INFERENCE_EHR_TYPE in ['mimic', 'eicu', 'p12']:
                if args.ehr_type != INFERENCE_EHR_TYPE:
                    globals()['{}_test_dataset'.format(INFERENCE_EHR_TYPE)] = TS_EHR_Dataset(data_dict[f'{INFERENCE_EHR_TYPE}_test'], args, 'test')
                else:
                    train_dataset = TS_EHR_Dataset(data_dict['train'], args, 'train')
                    valid_dataset = TS_EHR_Dataset(data_dict['valid'], args, 'valid')
                    globals()['{}_test_dataset'.format(INFERENCE_EHR_TYPE)] = TS_EHR_Dataset(data_dict['test'], args, 'test')
            
            # Dataloaders
            batch_size = args.batch_size

            # WeightedRandomSampler
            if args.oversampling:
                class_sample_counts = np.bincount(train_dataset.y_data)
                class_weights = 1.0 / class_sample_counts
                print('Class Weights',class_sample_counts, class_weights)
                sample_weights = class_weights[train_dataset.y_data]
                
                sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),  
                                                num_samples=len(sample_weights),             
                                                replacement=True)
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                            sampler=sampler, num_workers=0)
            else:
                train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                            shuffle=False, num_workers=0)

            valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                                    shuffle=False, num_workers=0)
            
            mimic_test_loader = torch.utils.data.DataLoader(dataset=mimic_test_dataset, batch_size=batch_size, 
                                                            shuffle=False, num_workers=0)
            
            eicu_test_loader = torch.utils.data.DataLoader(dataset=eicu_test_dataset, batch_size=batch_size, 
                                                            shuffle=False, num_workers=0)
            
            p12_test_loader = torch.utils.data.DataLoader(dataset=p12_test_dataset, batch_size=batch_size, 
                                                            shuffle=False, num_workers=0)
            
            return train_loader, valid_loader, mimic_test_loader, eicu_test_loader, p12_test_loader

    else:
        train_dataset = TS_EHR_Dataset(data_dict['train_spectrum'], args, 'train')
        valid_dataset = TS_EHR_Dataset(data_dict['valid_spectrum'], args, 'valid')
        test_dataset = TS_EHR_Dataset(data_dict['test_spectrum'], args, 'test')

        # Dataloaders
        batch_size = args.batch_size

        # WeightedRandomSampler
        if args.oversampling:
            class_sample_counts = np.bincount(train_dataset.y_data)
            class_weights = 1.0 / class_sample_counts
            print('Class Weights',class_sample_counts, class_weights)
            sample_weights = class_weights[train_dataset.y_data]
            
            sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),  
                                            num_samples=len(sample_weights),             
                                            replacement=True)
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                        sampler=sampler, num_workers=0)
        else:
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                                        shuffle=False, num_workers=0)

        valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=0)
        
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                                shuffle=False, num_workers=0)
        
        return train_loader, valid_loader, test_loader
        
    # spectrum_group1_dataset = TS_EHR_Dataset(data_dict['spectrum_group1'], args, 'spectrum_group1')
    # spectrum_group2_dataset = TS_EHR_Dataset(data_dict['spectrum_group2'], args, 'spectrum_group2')
    # spectrum_group3_dataset = TS_EHR_Dataset(data_dict['spectrum_group3'], args, 'spectrum_group3')
    # spectrum_group4_dataset = TS_EHR_Dataset(data_dict['spectrum_group4'], args, 'spectrum_group4')

    
    # spectrum_group1_loader = torch.utils.data.DataLoader(dataset=spectrum_group1_dataset, batch_size=batch_size,
    #                                                      shuffle=False, num_workers=0)
    
    # spectrum_group2_loader = torch.utils.data.DataLoader(dataset=spectrum_group2_dataset, batch_size=batch_size,
    #                                                      shuffle=False, num_workers=0)
    
    # spectrum_group3_loader = torch.utils.data.DataLoader(dataset=spectrum_group3_dataset, batch_size=batch_size,
    #                                                      shuffle=False, num_workers=0)
    
    # spectrum_group4_loader = torch.utils.data.DataLoader(dataset=spectrum_group4_dataset, batch_size=batch_size,
    #                                                      shuffle=False, num_workers=0)
    
    # return train_loader, valid_loader, test_loader, spectrum_group1_loader, spectrum_group2_loader, spectrum_group3_loader, spectrum_group4_loader