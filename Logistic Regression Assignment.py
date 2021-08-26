#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import statsmodels.api as sm
from patsy import dmatrices
import scipy.stats as stats
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


affairs= sm.datasets.fair.load_pandas().data


# In[3]:


affairs.head()


# In[4]:


len(affairs)


# In[5]:


affairs.isnull().sum()


# In[6]:


affairs['affair']=np.where(affairs.affairs>0,1,0)


# In[7]:


affairs


# In[8]:


plt.figure(figsize=(20,20))
plot=1
for i in affairs.columns:
    if plot<=9:
        ax=plt.subplot(3,3,plot)
        sns.distplot(affairs[i])
        plt.xlabel(i)
    plot+=1
plt.tight_layout()


# In[9]:


for i in affairs.columns:
    print(i,affairs[i].nunique())


# In[10]:


affairs.affair.value_counts()


# In[11]:


for i in affairs.drop('affair',axis=1).columns:
    sns.countplot(affairs[i],hue=affairs.affair)
    plt.xlabel(i)
    plt.show()


# In[12]:


plt.figure(figsize=(12,4))
sns.heatmap(affairs.corr(),annot=True)


# In[13]:


affairs['married_long']=np.where(affairs.yrs_married>6,1,0)


# In[14]:


affairs['young']=np.where(affairs.age<32,1,0)


# In[15]:


val_rate=affairs['rate_marriage'].unique()


# In[16]:


val_rate


# In[17]:


for val in val_rate:
    affairs['rate_'+str(val)]=np.where(affairs['rate_marriage']==val,1,0)


# In[18]:


affairs


# In[19]:


affairs.drop('rate_marriage',axis=1,inplace=True)


# In[20]:


X=affairs.drop('affair',axis=1)
y=affairs['affair']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=7)


# In[21]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_train_scaled=scale.fit_transform(x_train)
x_test_scaled=scale.transform(x_test)


# In[22]:


x_train_scaled


# In[24]:


import pickle
pickle.dump(scale,open('log_scale.pickle','wb'))


# In[31]:


y=np.exp(0)
y


# In[33]:


y_train_scaled=scale.fit_transform(y_train.values.reshape(-1,1))
y_test_scaled=scale.transform(y_test.values.reshape(-1,1))


# In[ ]:





# In[34]:


model=LogisticRegression()
model.fit(x_train_scaled,np.ravel(y_train_scaled.astype('int64'),order="C"))


# In[35]:


pred=model.predict(x_test_scaled)


# In[36]:


from sklearn.metrics import confusion_matrix
print(confusion_matrix(np.ravel(y_test_scaled.astype('int64'),order="C"),pred))


# In[37]:


from sklearn.metrics import roc_auc_score
auc = roc_auc_score(pred,np.ravel(y_test_scaled.astype('int64'),order="C"))
auc


# In[38]:


from sklearn.model_selection import GridSearchCV
log=LogisticRegression()
grid_values = { 'C': [0.001,0.01,0.1,1,10,100,1000]}
grid = GridSearchCV(estimator=model, param_grid=grid_values)


# In[39]:


grid.fit(x_train_scaled,np.ravel(y_train_scaled.astype('int64'),order="C"))


# In[40]:


pred_grid=grid.predict(x_test_scaled)


# In[41]:


print(confusion_matrix(np.ravel(y_test_scaled.astype('int64'),order="C"),pred_grid))


# In[42]:


from sklearn.metrics import classification_report
print(classification_report(np.ravel(y_test_scaled.astype('int64'),order="C"),pred_grid))


# In[43]:


grid.best_params_


# In[44]:


log=LogisticRegression(C=1000)
log.fit(x_train_scaled,np.ravel(y_train_scaled.astype('int64'),order="C"))


# In[45]:


final_pred=log.predict(x_test_scaled)


# In[46]:


print(confusion_matrix(np.ravel(y_test_scaled.astype('int64'),order="C"),final_pred))


# In[ ]:




