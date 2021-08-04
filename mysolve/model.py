from transformers import GPT2LMHeadModel, GPT2Config
import torch
import torch.nn as nn
from torch.nn import functional as F
def change_model(model,size=0):
    wte=model.transformer.wte.weight
    t=nn.Embedding(wte.shape[0]+size,wte.shape[1])
    t.weight.data=torch.cat([wte.data,torch.randn(size,wte.shape[1])])
    model.transformer.wte=t
    return model.transformer

class Net(nn.Module):
    def __init__(self,config_path,model_path,output_features,clip=1e-1):
        super(Net,self).__init__()
        self.output_features=output_features
        self.clip=clip
        self.config=GPT2Config.from_json_file(config_path)
        self.model=GPT2LMHeadModel(config=self.config)
        self.weight=torch.load(model_path)
        self.model.load_state_dict(self.weight)
        self.model=change_model(self.model)
        self.linear=nn.Linear(768,self.output_features)
    def forward(self,x):
        x=self.model(x)[0][...,-1,:]
        x=self.linear(x)
        x=F.softmax(x,dim=-1)
        return x/x.max(dim=-1).values.view(*x.shape[:-1],1)*(1-2*self.clip)+self.clip
        
if __name__=='__main__':
    model=Net(config_path="mysolve/config/model_config.json",
        model_path='mysolve\pytorch_model.bin',
        output_features=1024
    )
    output=model(torch.tensor([[1,2,3,4]]))