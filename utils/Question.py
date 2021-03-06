r'''
Author       : PiKaChu_wcg
Date         : 2021-08-07 19:04:06
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-11 21:17:52
FilePath     : \ifly\utils\Question.py
'''
import torch
from torch.utils.data import Dataset
class Question(Dataset):
    def __init__(self,data,feature_nums):
        self.data=data
        self.keys=list(data.keys())
        self.feature_nums=feature_nums
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index) :
        idx=self.keys[index]
        line=self.data[idx]['line']
        line=torch.tensor(line, dtype=torch.long)
        k_id=[torch.tensor([self.data[idx][i]]) for i in range(1,self.feature_nums+1)]
        q=torch.tensor([self.data[idx]["q_Level"]])
        # print(idx)
        return line,*k_id,q
class Question_test(Dataset):
    def __init__(self,data):
        self.data=data
        self.keys=list(self.data.keys())
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        idx=self.keys[index]
        l=self.data[idx]["k_level"]
        t=torch.tensor(self.data[idx]['line'])
        return t,idx,l