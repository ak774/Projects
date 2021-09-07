#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot  as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


titanic=pd.read_csv('https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv')


# In[3]:


titanic.head()


# In[4]:


(titanic.isnull().sum()/len(titanic))*100


# In[5]:


len(titanic)


# In[6]:


titanic.dtypes


# In[7]:


num_var=[feat for feat in titanic.columns if titanic[feat].dtype!='O' and feat!='PassengerId']
num_var


# In[8]:


cat_var=[feat for feat in titanic.columns if feat not in num_var]
cat_var


# In[9]:



for feat in num_var :
    plt.figure(figsize=(12,5))
    sns.histplot(x=feat,data=titanic,hue='Survived')
    plt.xlabel(feat)
    plt.ylabel('Survived')
    plt.show()


# In[10]:


sns.heatmap(titanic.drop('PassengerId',axis=1).corr(),annot=True)
plt.figure(figsize=(12,5))


# In[11]:


plt.figure(figsize=(12,5))
sns.histplot(x='Sex',data=titanic,hue='Survived')


# In[12]:


titanic.drop('PassengerId',axis=1,inplace=True)


# In[13]:


titanic['Sex'].value_counts()


# In[14]:


import scipy.stats as stats
import pylab
for feat in num_var:
    plt.figure(figsize=(12,5))
    sns.distplot(titanic[feat])
    plt.xlabel(feat)
    plt.show()


# In[15]:


for feat in num_var:
    plt.figure(figsize=(9,4))
    plt.subplot(1,2,1)
    titanic[feat].hist()
    plt.subplot(1,2,2)
    stats.probplot(titanic[feat],dist='norm',plot=pylab)
    plt.show()


# # Imputation

# In[16]:


titanic['Age']=titanic['Age'].fillna(titanic.Age.median())


# In[17]:


titanic['Cabin']=titanic['Cabin'].replace(np.NaN,'M')


# In[18]:


titanic.head()


# In[19]:


titanic['Age'].value_counts()


# In[20]:


titanic.isnull().sum()


# In[21]:


titanic.dropna(inplace=True)


# In[22]:


len(titanic)


# # Preprocessing

# In[23]:


titanic['has_sibling']=titanic['SibSp'].apply(lambda x:1 if x>0 else 0)


# In[24]:


titanic['has_child']=titanic['Parch'].apply(lambda x:1 if x>0 else 0)


# In[25]:


titanic['Fare'].describe()


# In[26]:


titanic['Fare'].sort_values().value_counts()


# In[27]:


titanic['is_not_cheap']=titanic['Fare'].apply(lambda x:1 if x>26 else 0)


# In[28]:


titanic['Cabin']=titanic['Cabin'].apply(lambda x:x[0] )


# In[29]:


titanic=pd.concat([titanic,pd.get_dummies(titanic['Sex'],drop_first=True)],axis=1)
titanic


# In[30]:


titanic.drop(['Sex'],axis=1,inplace=True)


# In[31]:


titanic.drop(['Ticket'],axis=1,inplace=True)


# In[32]:


titanic


# In[33]:


titanic['Age'].value_counts()


# In[34]:


titanic['Title']=(titanic['Name'].apply(lambda x:x.split(" ")[1])).apply(lambda x:x.split(".")[0])


# In[35]:


plt.figure(figsize=(25,15))
sns.histplot(x='Title',data=titanic,hue='Survived')


# In[36]:


titanic['is_mr']=titanic['Title'].apply(lambda x:1 if x=='Mr' else 0)


# In[37]:


titanic.drop(['Name'],axis=1,inplace=True)


# In[38]:


titanic['Embarked'].value_counts()


# In[39]:


titanic['Cabin'].value_counts()


# In[40]:


plt.figure(figsize=(25,15))
sns.histplot(x='Cabin',data=titanic,hue='Survived')


# In[41]:


titanic['is_missing']=titanic['Cabin'].apply(lambda x:0 if x=='M' else 0)
titanic.drop(['Cabin'],axis=1,inplace=True)


# In[42]:


titanic


# In[43]:


titanic=pd.concat([titanic,pd.get_dummies(titanic['Embarked'],drop_first=True)],axis=1)
titanic


# In[44]:


titanic.drop(['Title'],axis=1,inplace=True)


# In[45]:


titanic.drop(['Embarked'],axis=1,inplace=True)


# # model

# In[46]:


X=titanic.drop(['Survived'],axis=1)
y=titanic['Survived']


# In[47]:


X


# In[48]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[49]:


tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)


# In[50]:


pred=tree.predict(x_test)


# In[51]:


print(confusion_matrix(y_test,pred))


# In[52]:


print(classification_report(y_test,pred))


# In[53]:


from sklearn.model_selection import GridSearchCV
param = {
    'criterion': ['gini', 'entropy'],
    'max_depth' : range(2,32,1),
    'min_samples_leaf' : range(1,10,1),
    'min_samples_split': range(2,10,1),
    'splitter' : ['best', 'random']
    
}
tree1=DecisionTreeClassifier()
grid=GridSearchCV(tree1,param,cv=5)
grid.fit(x_train,y_train)


# In[54]:


pred_grid=grid.predict(x_test)


# In[55]:


print(confusion_matrix(y_test,pred_grid))


# In[56]:


grid.best_params_


# In[57]:


tree1=DecisionTreeClassifier(criterion='entropy',max_depth =9,
  min_samples_leaf= 6,
  min_samples_split= 9,
  splitter= 'random')


# In[58]:


tree1.fit(x_train,y_train)


# In[59]:


pred2=tree1.predict(x_test)


# In[60]:


print(confusion_matrix(y_test,pred2))


# In[61]:


import pickle
pickle.dump(tree1,open('tree_fin.pickle','wb'))

