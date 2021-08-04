from transformers import BertTokenizerFast
import pandas as pd
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np

data_path="t.csv"
vocab_path=r'vocab\vocab.txt'

def collate_fn(batch):
    # print("batch:{}".format(batch))
    lines=[line[0] for line in batch]
    # print(f"lines:{line}")
    input_ids = rnn_utils.pad_sequence(lines, batch_first=True, padding_value=0)
    pos=torch.cat([i[1].view(1,*i[1].shape) for i in batch],dim=0)
    neg=torch.cat([i[2].view(1,*i[2].shape) for i in batch],dim=0)
    return input_ids,pos,neg

class Question(Dataset):
    def __init__(self,data,features_num):
        self.features_num=features_num
        self.data=data
        self.keys=list(data.keys())
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index) :
        idx=self.keys[index]
        line=self.data[idx]['line']
        line=torch.tensor(line, dtype=torch.long)
        pos_label=torch.zeros(self.features_num)
        pos_label[self.data[idx]['pos_label']]=1
        neg_label_p=np.random.randint(low=0,high=self.features_num,size=len(self.data[idx]["pos_label"]))
        neg_label=torch.zeros(self.features_num)
        neg_label[neg_label_p]=1
        return line,pos_label,neg_label

def preprocess(data_path=data_path,vocab_path=vocab_path,batch_size=2,features_nums=1295):
    df=pd.read_csv(data_path)
    tokenizer=BertTokenizerFast(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    dict={}
    for _,item in df.iterrows():
        if(item.TestQuestionID not in dict.keys()):
            line=[cls_id]
            if(item.type):
                line+=tokenizer.encode(item.type[0],add_special_tokens=False)
            line+=[sep_id]
            if(item.Content):
                line+=tokenizer.encode(item.Content,add_special_tokens=False)
            line+=[sep_id]
            if type(item.Analysis)==str:
                line+=tokenizer.encode(item.Analysis,add_special_tokens=False)
            line+=[sep_id]
            if item.options:
                line+=tokenizer.encode(item.options,add_special_tokens=False)
            line+=[sep_id]
            dict[item.TestQuestionID]={}
            dict[item.TestQuestionID]['line']=line
            dict[item.TestQuestionID]['pos_label']=[]
        dict[item.TestQuestionID]['pos_label'].append(item.KnowledgeID)
    data=Question(dict,features_nums)
    dataloader=DataLoader(
        data, batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        collate_fn=collate_fn
    )
    return dataloader

if __name__=='__main__':
    preprocess()