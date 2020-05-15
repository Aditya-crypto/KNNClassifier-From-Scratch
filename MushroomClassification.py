#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from collections import Counter
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# In[5]:


class KNNClassifier:
    
    
    def __init__(self):
        pass
    
    def count1(self,List): 
        occurence_count = Counter(List) 
        return occurence_count.most_common(1)[0][0]
    
    def preprocessdata(self,df):
        global switcher
        df1=df.iloc[:,:].values
        columns=list(df1[:,11])
        val=self.count1(columns)
        val1=ord(val)-97
        switcher={
         'a':0,'b':1,'c':2,'d':3,'e':4,'f':5,'g':6,'h':7,
          'i':8,'j':9,'k':10,'l':11,'m':12,'n':13,'o':14,'p':15,'q':16,'r':17,'s':18,'t':19,
           'u':20,'v':21,'w':22,'x':23,'y':24,'z':25,'?':val1,
       }
        df=df.applymap(lambda x: switcher.get(x))
        return df
        
    def train(self,path):
        df = pd.read_csv(path,header=None)
        df=self.preprocessdata(df)
        global train_frame
        global train_labels
        train_frame=df.iloc[:,1:].values
        train_labels=df.iloc[:,0].values
    
    def predict(self,path):
        test_frame=pd.read_csv(path,header=None)
        test_frame=self.preprocessdata(test_frame)
        predlist=self.EucledianDist(train_frame,test_frame,train_labels,3)
        df=pd.DataFrame(predlist)
        df=df.applymap(lambda x:x+97)
        df=df.applymap(chr)
        df=df.values.tolist()
        print(df)
        return df
        
    def EucledianDist(self,train_frame,test_frame,train_labels,k):
        predicted_label=[]
        print(test_frame)
        for i in test_frame.iloc[:,:].values:
            result=[]
            index=0
            for j in train_frame:
                eudist=0.0
                val=np.linalg.norm(i-j)
                result.append((val,int(train_labels[index])))
                index+=1
            result.sort()
            klist=[]
            for p in range(0,k):
                klist.append(result[p][1])
            resval=self.count1(klist)
            print(resval)
            predicted_label.append(resval)
        return predicted_label


# In[ ]:




