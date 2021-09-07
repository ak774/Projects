#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
import scipy.stats as stats
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=load_boston()
boston=pd.DataFrame(df.data,columns=df.feature_names)


# In[3]:


boston.head()


# In[4]:


target=pd.DataFrame(df.target,columns=['Target'])


# In[5]:


target


# In[6]:


boston.isnull().sum()


# In[7]:


boston.dtypes


# In[8]:


plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in boston:
    if plotnumber<=13 :
        ax = plt.subplot(4,4,plotnumber)
        sns.distplot(boston[column])
        plt.xlabel(column,fontsize=20)
        
    plotnumber+=1
plt.tight_layout()


# In[9]:


import pylab
def plot_data(df,feature):
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    df[feature].hist()
    plt.subplot(1,2,2)
    stats.probplot(df[feature],dist='norm',plot=pylab)
    plt.show()
for column in boston:
    plot_data(boston,column)


# In[10]:


for i in boston.columns:
    plt.scatter(boston[i],target)
    plt.ylabel('target')
    plt.xlabel(i)
    plt.show()


# In[11]:



sns.pairplot(boston)


# In[12]:


plt.figure(figsize=(12,4))
sns.heatmap(boston.corr(),annot=True)


# In[13]:


len(boston)


# In[14]:


col=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'AGE', 'DIS', 'RAD', 'TAX',
       'PTRATIO', 'B']
for i in col:
    boston[i]=np.log(boston[i], out=np.zeros_like(boston[i]), where=(boston[i]!=0))


# In[15]:


boston


# In[16]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(boston,target,test_size=0.2)


# In[17]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_train_scale=scale.fit_transform(x_train)
x_test_scale=scale.transform(x_test)


# In[18]:


y_train_scale=scale.fit_transform(y_train)
y_test_scale=scale.transform(y_test)


# In[19]:


model=LinearRegression()
model.fit(x_train_scale,y_train_scale)


# In[20]:


pred=model.predict(x_test_scale)


# In[21]:


plt.scatter(pred,y_test_scale)


# In[22]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test_scale,pred)


# In[25]:


from sklearn.linear_model import RidgeCV,Ridge
alphas = np.random.uniform(low=0, high=10, size=(50,))
ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)
ridgecv.fit(x_train_scale, y_train_scale)


# In[26]:


ridge=Ridge(alpha=ridgecv.alpha_)
ridge.fit(x_train_scale,y_train_scale)


# In[27]:


ridge.score(x_test_scale, y_test_scale)


# In[28]:


pred1=ridge.predict(x_test_scale)


# In[29]:


mean_squared_error(y_test_scale,pred1)


# In[30]:


plt.scatter(pred1,y_test_scale)


# In[32]:


import pickle
filename = 'final_model.pickle'
pickle.dump(ridge, open(filename, 'wb'))


# In[ ]:




