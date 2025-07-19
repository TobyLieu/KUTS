import pandas as pd
import numpy as np
import pickle
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-base")
    
class MyDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        with open(self.file_path, 'rb') as f:
            self.data = pickle.load(f)

    def save_data(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.data, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        label = self.data[item]['label']
        cc = self.data[item]['cc']
        age = self.data[item]['age']
        gender = self.data[item]['gender']
        dp = self.data[item]['dp']
        sp = self.data[item]['sp']
        sense = self.data[item]['sense']
        temp = self.data[item]['temp']
        spo = self.data[item]['spo']
        breath = self.data[item]['breath']
        hr = self.data[item]['hr']
        lai = self.data[item]['lai']
        text = self.data[item]['english_text']
        
        # grade_rec = classify_medical_conditions(temp, hr, breath, sp, spo)

        return label, cc, age, gender, dp, sp, sense, temp, spo, breath, hr, lai, text
    
    def remove_item(self, index_list):
        for index in index_list:
            if 0 <= index < len(self.data):
                del self.data[index]
            else:
                print("Index out of range.")
                
        # self.save_data()
    
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py

        batch_label, batch_cc, batch_age, batch_gender, batch_dp, batch_sp, batch_sense, batch_temp, batch_spo, batch_breath, batch_hr, batch_lai, batch_text = tuple(zip(*batch))

        batch_label = torch.as_tensor(batch_label)
        batch_age = torch.as_tensor(batch_age)
        batch_gender = torch.as_tensor(batch_gender)
        batch_dp = torch.as_tensor(batch_dp)
        batch_sp = torch.as_tensor(batch_sp)
        batch_sense = torch.as_tensor(batch_sense)
        batch_temp = torch.as_tensor(batch_temp)
        batch_spo = torch.as_tensor(batch_spo)
        batch_breath = torch.as_tensor(batch_breath)
        batch_hr = torch.as_tensor(batch_hr)
        batch_lai = torch.as_tensor(batch_lai)
        # batch_grade_rec = torch.as_tensor(batch_grade_rec)
        
        bert_out = tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_text,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=160,
                                       return_tensors='pt',
                                       return_length=True)
        # input_ids:编码之后的数字
        # attention_mask:是补零的位置是0,其他位置是1
        input_ids = bert_out['input_ids']
        attention_mask = bert_out['attention_mask']
        token_type_ids = bert_out['attention_mask']

        return batch_label, batch_age, batch_gender, batch_dp, batch_sp, batch_sense, batch_temp, batch_spo, batch_breath, batch_hr, batch_lai, input_ids, attention_mask, token_type_ids, batch_text


class MIMICDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.load_data()

    def load_data(self):
        with open(self.file_path, 'rb') as f:
            self.data = pickle.load(f)
            
    def save_data(self):
        with open(self.file_path, 'wb') as f:
            pickle.dump(self.data, f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        label = self.data[item]['label']
        cc = self.data[item]['cc']
        age = self.data[item]['age']
        gender = self.data[item]['gender']
        dp = self.data[item]['dp']
        sp = self.data[item]['sp']
        sense = self.data[item]['pain']
        temp = self.data[item]['temp']
        spo = self.data[item]['spo']
        breath = self.data[item]['breath']
        hr = self.data[item]['hr']
        lai = self.data[item]['lai']
        # text = self.data[item]['text']
        text = self.data[item]['know_text']
        
        # grade_rec = classify_medical_conditions(temp, hr, breath, sp, spo)

        return label, cc, age, gender, dp, sp, sense, temp, spo, breath, hr, lai, text
    
    def remove_item(self, index_list):
        for index in index_list:
            if 0 <= index < len(self.data):
                del self.data[index]
            else:
                print("Index out of range.")
                
        # self.save_data()
    
    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py

        batch_label, batch_cc, batch_age, batch_gender, batch_dp, batch_sp, batch_sense, batch_temp, batch_spo, batch_breath, batch_hr, batch_lai, batch_text = tuple(zip(*batch))

        batch_label = torch.as_tensor(batch_label)
        batch_age = torch.as_tensor(batch_age)
        batch_gender = torch.as_tensor(batch_gender)
        batch_dp = torch.as_tensor(batch_dp)
        batch_sp = torch.as_tensor(batch_sp)
        batch_sense = torch.as_tensor(batch_sense)
        batch_temp = torch.as_tensor(batch_temp)
        batch_spo = torch.as_tensor(batch_spo)
        batch_breath = torch.as_tensor(batch_breath)
        batch_hr = torch.as_tensor(batch_hr)
        batch_lai = torch.as_tensor(batch_lai)
        # batch_grade_rec = torch.as_tensor(batch_grade_rec)
        
        bert_out = tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_text,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=160,
                                       return_tensors='pt',
                                       return_length=True)
        # input_ids:编码之后的数字
        # attention_mask:是补零的位置是0,其他位置是1
        input_ids = bert_out['input_ids']
        attention_mask = bert_out['attention_mask']
        token_type_ids = bert_out['attention_mask']

        return batch_label, batch_age, batch_gender, batch_dp, batch_sp, batch_sense, batch_temp, batch_spo, batch_breath, batch_hr, batch_lai, input_ids, attention_mask, token_type_ids, batch_text
    