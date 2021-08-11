r'''
Author       : PiKaChu_wcg
Date         : 2021-08-12 03:41:42
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-12 06:44:14
FilePath     : \ifly\submit.py
'''
import torch
import pandas as pd
import preprocess
from tqdm import tqdm
from preprocess import preprocess_test
import warnings
import numpy as np
warnings.filterwarnings('ignore')
class Submit:
    def __init__(self,data_path,model_path,kd_path,vocab_path):
        self.data=pd.read_csv(data_path)
        self.model=torch.load(model_path).cpu()
        self.kd=torch.load(kd_path)
        self.dataloader=preprocess_test(self.data,vocab_path)
        self.result=pd.read_csv(data_path)[['index','TestQuestionID']]
    def mkres(self,to):
        for _,item in tqdm(enumerate(self.dataloader)):
            input=item[0]
            k_level=item[2][0]
            testid=item[1][0]
            try:
                output=self.model(input)
            except:
                print(output)
                continue
            o=[]
            for i in range(len(self.model.slice())-1):
                o.append(output[...,self.model.slice()[i]:self.model.slice()[i+1]])
            kid=o[k_level-1].argmax(dim=-1).item()
            kid=self.kd.check_l(k_level,kid)
            self.result.loc[self.result['TestQuestionID']==testid,'KnowledgeID']=kid
            self.result.loc[self.result['TestQuestionID']==testid,'q_Level']=o[-1].argmax(dim=-1).item()+1
            self.result=self.result.fillna(0)
            self.result['KnowledgeID']=self.result['KnowledgeID'].astype(np.int64)
            self.result['q_Level']=self.result['q_Level'].astype(np.int64)
        self.result.to_csv(to,index=False)
        return self.result



import os 
for root,dirs,files in os.walk("model/exp1/"): 
    for file in files:
        s=Submit('data/test_data.csv',"model/exp1/"+file,'model/kd.pth','vocab/vocab.txt')  
        s.mkres('res/exp1/'+file.split(".")[0]+".csv")