r'''
Author       : PiKaChu_wcg
Date         : 2021-08-04 21:46:14
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-07 01:16:56
FilePath     : \mysolve\train.py
'''
from tensorboardX import SummaryWriter
from preprocess import preprocess
from model import Net
import torch
from tqdm import tqdm
from IPython.display import  clear_output  
clear_output()
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
batch_size=2
features_num=1295
data_path='data/train_data.csv'
vocab_path='vocab/vocab.txt'
config_path="config/model_config.json"
model_path='model/GPT2_transformer.pth'
clip=1e-1
epoche=10
use_gpu=torch.cuda.is_available()
# use_gpu=False
writer=SummaryWriter("runs/exp1")

dataloader=preprocess(
    data_path=data_path,
    vocab_path=vocab_path,
    features_nums=features_num,
    batch_size=batch_size
)

model=Net(
    config_path=config_path,
    model_path=model_path,
    output_features=features_num
)
if use_gpu:
    model=model.cuda()
model.change_param_state("model")

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
for e in range(30,33):
    for batch,(line,pos,neg) in tqdm(enumerate(dataloader)):
        model.train()
        if(use_gpu):
            line=line.cuda()
            pos=pos.cuda()
            neg=neg.cuda()
        output=model(line)
        loss_pos=-pos*torch.log(output)
        loss_neg=-neg*torch.log(1-output)
        loss=loss_pos+loss_neg
        loss=loss.sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch==0:
            if use_gpu:
                writer.add_scalar("train_loss",loss.cpu(),e)
            else:
                writer.add_scalar("train_loss",loss,e)