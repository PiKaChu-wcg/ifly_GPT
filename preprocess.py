r'''
Author       : PiKaChu_wcg
Date         : 2021-08-01 07:20:56
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-12 03:27:55
FilePath     : \ifly\preprocess.py
'''
import torch
from utils.KnowledgeDict import KnowledgeDict
from utils.Question import Question,Question_test as qt
import pandas as pd
from transformers import BertTokenizerFast
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
data_path="data/train_data.csv"
vocab_path="vocab/vocab.txt"
def preprocess(data_path=data_path,vocab_path=vocab_path,batch_size=2,k_level=3):
    df=pd.read_csv(data_path)
    KD=KnowledgeDict(3,df)
    dict={}
    tokenizer=BertTokenizerFast(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
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
            dict[item.TestQuestionID]['q_Level']=item.q_Level-1
        dict[item.TestQuestionID][item.k_Level]=KD.check_k(item.k_Level,item.KnowledgeID)
    dataset=Question(dict,k_level) 
    def collate_fn(batch):
        # print("batch:{}".format(batch))
        lines=[line[0][:1024] for line in batch]
        t=[]
        for k in range(1,len(batch[0])):
            t.append(torch.cat([i[k] for i in batch]))
        # print(f"lines:{line}")
        input_ids = rnn_utils.pad_sequence(lines, batch_first=True, padding_value=0)

        t.insert(0,input_ids)
        return t
    dataloader=DataLoader(
        dataset, batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        collate_fn=collate_fn
    )  
    return dataloader,KD,dict
if __name__=="__main__":
    _,kd,__=preprocess()
    torch.save(kd,"model/kd.pth")
data_path="data/test_data.csv"
df=pd.read_csv(data_path)
vocab_path='vocab/vocab.txt'
def preprocess_test(df,vocab_path):
    """处理测试集数据

    Args:
        df (dataframe): 输入的数据
        vocab_path (str): 词汇路径
        batch_size ([int]): batchsize

    Returns:
        [Dataloader]]: return a data with type of dataloader
    """
    dict={}
    tokenizer=BertTokenizerFast(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
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
            dict[item.TestQuestionID]['k_level']=item.k_Level
    dataset=qt(dict) 
    def collate_fn(batch):
        lines=[line[0] for line in batch]
        idx=[i[1] for i in batch]
        l=[i[2]for i in batch]
        input_ids = rnn_utils.pad_sequence(lines, batch_first=True, padding_value=0)
        return input_ids,idx,l
    dataloader=DataLoader(
        dataset, batch_size=1, 
        shuffle=True, 
        drop_last=True,
        collate_fn=collate_fn
    )
    return dataloader
    

if __name__=="__main__":
    t=preprocess_test(df,vocab_path,2)
    print(next(iter(t)))