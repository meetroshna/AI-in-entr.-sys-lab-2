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


# # random forest 

# In[10]:


rfc=RandomForestClassifier(n_estimators=60,random_state=0)
rfc.fit(X_train,Y_train)

Y_pred=rfc.predict(X_test)

cm=confusion_matrix(Y_pred,Y_test)
class_label = ["malignant", "benign"]
df_cm = pd.DataFrame(cm, index=class_label,columns=class_label)

precision, recall, fscore, train_support = score(Y_test, Y_pred, pos_label=1, average='binary')
Random_forest_classifier_accuracy=round(acs(Y_test,Y_pred), 4)*100
print('Precision: {} \nRecall: {} \nF1-Score: {} \nAccuracy: {}'.format(
    round(precision, 3), round(recall, 3), round(fscore,3), Random_forest_classifier_accuracy) +"% \n")

sns.heatmap(df_cm,annot=True,cmap='Pastel1',linewidths=2,fmt='d')
plt.title("Confusion Matrix",fontsize=15)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

