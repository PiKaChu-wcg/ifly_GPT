r'''
Author       : PiKaChu_wcg
Date         : 2021-08-07 03:32:07
LastEditors  : PiKachu_wcg
LastEditTime : 2021-08-07 18:39:30
FilePath     : \ifly\utils\KnowledgeDict.py
'''
from typing import List, Tuple
import pandas as pd
class KnowledgeDict:
    """一个记录知识点和(知识等级,序号)对应关系的类,有feature_size能返回各个知识等级有的知识点的数目
    """
    def __init__(self,k_Level,df=None):
        self.kl=[]
        self.kid2kl={}
        for i in range(k_Level+1):
            self.kl.append([])
        if  isinstance(df,pd.DataFrame):
            self.load_df(df)
    def add_k(self,kid:int,k_level:int):
        """增加一个新的知识点

        Args:
            kid ([int]): 知识点id
            k_level ([int]): 知识点等级
        """
        if kid not in self.kid2kl.keys():
            self.kl[k_level].append(kid)    
            self.kid2kl[kid]=(k_level,len(self.kl[k_level])-1)
    def check_l(self,k_level:int,id:int)->int:
        """查看一个指定k_level的第id个元素的knowledge id

        Args:
            k_level (int): [description]
            id (int): [description]

        Returns:
            int: [description]
        """
        return self.kl[k_level][id]
    def check_k(self,kid:int)->Tuple:
        """查看一个knowledgeid对应的k_level和序号

        Args:
            kid (int): [description]

        Returns:
            Tuple: [description]
        """
        return self.kid2kl[kid]
    def feature_size(self)->List[int]:
        """等到各个k_level的元素与个数

        Returns:
            List[int]: [description]
        """ 
        return [len(i) for i in self.kl[1:]]
    def load_df(self,df):
        for _,item in df.iterrows():
            self.add_k(item.KnowledgeID,item.k_Level)

if __name__=="__main__":
    df=pd.read_csv("../data/train_data.csv")
    KD=KnowledgeDict(2,df)