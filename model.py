r'''
Author       : PiKaChu_wcg
Date         : 2021-08-03 21:46:13
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-07 01:22:00
FilePath     : \mysolve\model.py
'''


from torch.nn.modules.linear import Linear
from transformers import GPT2LMHeadModel, GPT2Config
import torch
import torch.nn as nn
from torch.nn import functional as F

class Net(nn.Module):
    def __init__(self,model_path,output_features,clip=1e-1):
        """这是使用cpm作为预训练的GTP2模型,其中model是transformer块,再用最后的结果经过一个线性神经网络得到我们要的分类

        Args:
            config_path ([str]]): str
            model_path ([type]): [description]
            output_features ([type]): [description]
            clip ([type], optional): [description]. Defaults to 1e-1.
        """
        super(Net,self).__init__()
        self.output_features=output_features
        self.clip=clip
        self.model=torch.load(model_path)
        self.linear=nn.Sequential(
            nn.Linear(768,(output_features+2*768)),
            nn.ReLU(),
            nn.Linear((2*output_features+768),3*output_features),
            nn.ReLU(),
            nn.Linear((2*output_features+768),3*output_features),
            )
    def forward(self,x):
        x=self.model(x)[0][...,-1,:]
        x=self.linear(x)
        x=x.view(*x.shape[:-1],3,self.output_features)
        x=F.softmax(x,dim=-1)
        return x/x.max(dim=-1).values.view(*x.shape[:-1],1)*(1-2*self.clip)+self.clip
    def change_param_state(self,net:str="backbone",is_freeze:bool=True):
        """冻结神经网络的指定参数

        Args:
            net (str): 选择的参数:可选 backbone,linear,all
        """
        if net=="backbone":
            for param in self.model.parameters():
                param.requires_grad = not is_freeze
        if net=="linear":
            for param in self.model.parameters():
                param.requires_grad = not is_freeze
        if net=='all':
            for param in self.parameters():
                param.requires_grad= not is_freeze
        
if __name__=='__main__':
    model=Net(config_path="mysolve/config/model_config.json",
        model_path='mysolve\pytorch_model.bin',
        output_features=1024
    )
    output=model(torch.tensor([[1,2,3,4]]))