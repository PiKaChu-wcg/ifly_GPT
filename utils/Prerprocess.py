r'''
Author       : PiKaChu_wcg
Date         : 2021-08-11 20:50:57
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-11 21:24:00
FilePath     : \ifly\utils\Prerprocess.py
'''
import pandas as pd
from transformers import BertTokenizerFast
from Question import Question_test as qt
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader
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