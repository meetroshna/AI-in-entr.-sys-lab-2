#!/usr/bin/env python
# coding: utf-8

# # LAB 2

# # SUBMITTED BY: ROSHNA BABU (100805012)

# In[1]:


import pandas as pd
pd.set_option('use_inf_as_na', True)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score as acs
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Load Dataset
data=pd.read_csv(r"C:\Users\roshn\Downloads\data.csv")
data.head()


# In[3]:


# find info of dataset
data.info()


# In[4]:


#New data with "M"=1 and "B"=0
data1=data.copy()
def classifier(data1):
    if data1["diagnosis"]=="M":
        return "1"
    else:
        return "0"
data1["diagnosis"] = data1.apply(classifier, axis=1) 
data1.replace([np.inf, -np.inf], np.nan, inplace=True) 
data1["diagnosis"]=pd.to_numeric(data1["diagnosis"],errors="coerce") 


# In[5]:


print(data1.columns,data.shape)
print(data1.info())
print(data1.describe().T)
print(data1.nunique())


# In[6]:


sns.set(style="whitegrid") 
print(data['diagnosis'].value_counts())
fig = plt.figure(figsize = (10,6))
sns.countplot('diagnosis', data=data, palette='gist_heat')
plt.show()


# # 

# In[7]:


X=data1.iloc[:,2:32]
Y=data1.iloc[:,1]

for col in X.columns:
    X[col][np.isinf(X[col])]=X[col].mean()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[8]:


selected_features=data1.corr().diagnosis.sort_values(ascending=False).iloc[1:21][::-1].index

X_train = X_train[selected_features]
X_test = X_test[selected_features]


# In[9]:


selected_features=data1.corr().diagnosis.sort_values(ascending=False).iloc[1:21][::-1].index

X_train = X_train[selected_features]
X_test = X_test[selected_features]


# # LOGISTIC REGRESSION

# In[15]:


#Script for Logistical Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix  


# In[16]:


for name,method in [('Logistic Regression', LogisticRegression(solver='liblinear',random_state=100))]: 
    method.fit(X_train,Y_train)
    predict = method.predict(X_test)
    print(confusion_matrix(Y_test,predict))  
    print(classification_report(Y_test,predict))


# In[ ]:




