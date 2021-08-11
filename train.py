r'''
Author       : PiKaChu_wcg
Date         : 2021-08-04 21:46:14
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-12 03:51:38
FilePath     : \ifly\train.py
'''
from tensorboardX import SummaryWriter
from preprocess import preprocess
from model import Net
import torch
from tqdm import tqdm
from IPython.display import  clear_output  
import torch.nn as nn
from IPython.core.interactiveshell import InteractiveShell 
clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
batch_size=2
data_path='data/train_data.csv'
vocab_path='vocab/vocab.txt'
model_path='model/exp1.pth'
q_level=5
epoch=30
use_gpu=torch.cuda.is_available()
writer=SummaryWriter("runs/exp1")
dataloader,KD,d=preprocess(
    data_path=data_path,
    vocab_path=vocab_path,
    batch_size=batch_size,
    k_level=3
)
model=torch.load(model_path)
if use_gpu:
    model=model.cuda()
for param in model.parameters():
    param.requires_grad=True
s=model.slice()
optimizer = torch.optim.SGD(model.parameters(), lr=5e-4)
loss_fn=nn.CrossEntropyLoss()
for e in range(epoch):
    err=[]
    for batch,item in tqdm(enumerate(dataloader)):
        _=model.train()
        input=[]
        if(use_gpu):
            for i in item:
                i=i.cuda()
                input.append(i)
        else :
            input=item
        output=model(input[0])
        loss=torch.cat([loss_fn(output[...,s[i]:s[i+1]],input[i+1]).view(1) for i in range(4)])
        loss=loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        err.append(loss.cpu().item())
    writer.add_scalar("loss",sum(err)/len(err),e)
    print(sum(err)/len(err))
    if e%5==0 and e>=10:
        torch.save(model,"model/exp1/modele"+str(e)+".pth")

