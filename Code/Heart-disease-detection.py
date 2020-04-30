#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


disease=pd.read_csv(r'C:\Users\HP\Downloads\framingham.csv')


# In[4]:


disease.drop(['education'],axis=1,inplace=True)


# In[5]:


disease.dropna(axis=1,inplace=True)


# In[6]:


disease.rename(columns={'male':'Sex_male'},inplace=True)


# In[7]:


disease.head()


# In[8]:


disease.TenYearCHD.value_counts()


# In[13]:


plt.figure(figsize = (7, 5)) 
sns.countplot(x ='TenYearCHD', data = disease,  
             palette ="coolwarm" ) 
plt.show() 


# In[14]:


X=np.asarray(disease[['age','Sex_male','currentSmoker','prevalentStroke','prevalentHyp','diabetes','sysBP','diaBP']])
y=np.asarray(disease['TenYearCHD'])


# In[16]:


X=preprocessing.StandardScaler().fit(X).transform(X)


# In[17]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4)


# In[18]:


print ('Train set:', X_train.shape,  y_train.shape) 
print ('Test set:', X_test.shape,  y_test.shape) 


# In[19]:


from sklearn.linear_model import LogisticRegression
Regressor=LogisticRegression()
Regressor.fit(X_train,y_train)
predictions=Regressor.predict(X_test)


# In[20]:


from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,predictions)
print(cm)


# In[23]:


plt.figure(figsize=(8,5))
sns.heatmap(cm,annot=True,fmt='d',cmap="Greens")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


# In[24]:


print(classification_report(y_test,predictions))


# In[27]:


from sklearn.metrics import jaccard_similarity_score
jaccard_similarity_score(y_test,predictions)


# In[ ]:





# In[ ]:




