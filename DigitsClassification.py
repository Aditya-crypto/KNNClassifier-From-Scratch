#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import math
from collections import Counter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# In[10]:


class KNNClassifier:
    
    def __init__(self):
        pass
        
    def train(self,path):
        df=pd.read_csv(path,header=None)
        #self.train_test_split(df,.2)
        global test_frame
        global train_frame
        global test_labels
        global train_labels
        train_frame=df.iloc[:,1:].values 
        train_labels=df.iloc[:,:1].values
        
    def predict(self,path):
        test_frame=pd.read_csv(path,header=None)
        predlist=self.eucledianDist(train_frame,test_frame,train_labels,3)
        return predlist
        
    def count1(self,List): 
        occurence_count = Counter(List) 
        return occurence_count.most_common(1)[0][0]
    
    def eucledianDist(self,train_frame,test_frame,train_labels,k):
        predicted_label=[]
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
            predicted_label.append(resval)
        return predicted_label



