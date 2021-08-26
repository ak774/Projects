#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import sqlite3
cnx = sqlite3.connect('C:\\Users\\akshay goel\\archive\\database.sqlite')
df = pd.read_sql_query("SELECT * FROM Player_Attributes", cnx)
df.head()


# In[3]:


df.iloc[:2500,:].to_csv('fifa.csv',index=False)


# In[4]:


len(df)


# In[5]:


df.isnull().sum()/len(df)


# In[6]:


df.dtypes


# In[7]:


num_var=[feat for feat in df.columns if df[feat].dtype!='O']


# In[8]:


len(num_var)


# In[9]:


cat_var=[feat for feat in df.columns if feat not in num_var]


# In[10]:


len(cat_var)


# In[11]:


plt.figure(figsize=(30,25))
sns.heatmap(df.corr(),annot=True)


# In[12]:


data=df.copy()
data.dropna(inplace=True)
len(data)


# In[13]:


len(data)/len(df)


# In[14]:


plt.figure(figsize=(20,10))
for feat in num_var:
    sns.scatterplot(x=feat,y='overall_rating',data=df)
    plt.xlabel(feat)
    plt.ylabel('overall_rating')
    plt.show()


# In[15]:


df['overall_rating'].value_counts()


# In[16]:



for feat in cat_var:
    if feat!='date':
        plt.figure(figsize=(20,10))
        sns.countplot(df[feat].dropna(),hue=df['overall_rating'])
        plt.xlabel(feat)
        plt.ylabel('count')
        plt.show()


# In[17]:



for feat in num_var:
    sns.boxplot(x=feat,data=df)
    plt.xlabel(feat)
    plt.show()


# In[18]:


threshold=.85
def correlation(data,threshold):
    col_corr=set()
    corr_matrix=data.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if(abs(corr_matrix.iloc[i,j])>threshold):
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[19]:


correlation(df.drop('overall_rating',axis=1).iloc[:,:],threshold)


# # Preprocessing

# In[20]:


df.drop(['id','player_fifa_api_id','player_api_id','date'],axis=1,inplace=True)


# In[21]:


df.isnull().sum()


# In[22]:


l=['preferred_foot','attacking_work_rate','defensive_work_rate']
for feat in l:
    print(f'{feat}:value={df[feat].unique()}')


# In[23]:


for feat in l:
    print(f'{feat}:%_top_value={sum(df[feat].value_counts()[:3])/len(df)}')


# In[24]:


feat_null=[]
for feat in num_var:
    if feat in df.columns and df[feat].isnull().sum()<1000:
        feat_null.append(feat)
feat_null


# In[25]:


for feat in feat_null:
    df[feat]=df[feat].fillna(df[feat].mean())


# In[26]:


df.isnull().sum()


# In[27]:


df.dropna(inplace=True)


# In[28]:


len(df)


# In[29]:


num_var_new=[feat for feat in df.columns if df[feat].dtype!='O']


# In[30]:


import scipy.stats as stats
import pylab
for feat in num_var_new:
    stats.probplot(df[feat],dist='norm',plot=pylab)
    plt.xlabel(feat)
    plt.show()


# In[31]:


l=['preferred_foot','attacking_work_rate','defensive_work_rate']
for feat in l:
    print(f'{feat}:value={df[feat].unique()}')


# In[32]:


df=pd.concat([df.drop(['preferred_foot'],axis=1),pd.get_dummies(df['preferred_foot'],drop_first=True)],axis=1)


# In[33]:


df


# In[34]:


df['attacking_work_rate'].unique()


# In[35]:


d={'medium':2, 'high':3, 'low':1, 'None':0, 'le':0, 'norm':0, 'stoc':0, 'y':0}
df['attacking_work_rate']=df['attacking_work_rate'].map(d)


# In[36]:


df['defensive_work_rate'].unique()


# In[37]:


d={'medium':2, 'high':3, 'low':1, '5':0, 'ean':0, 'o':0, '1':0, 'ormal':0, '7':0, '2':0,
       '8':0, '4':0, 'tocky':0, '0':0, '3':0, '6':0, '9':0, 'es':0}
df['defensive_work_rate']=df['defensive_work_rate'].map(d)


# In[38]:


df.dtypes


# In[39]:


df.isnull().sum()


# In[40]:


for feat in df.columns:
    df[feat]=np.sqrt(df[feat])


# In[41]:


import scipy.stats as stats
import pylab
for feat in df.columns:
    stats.probplot(df[feat],dist='norm',plot=pylab)
    plt.xlabel(feat)
    plt.show()


# In[42]:


for feat in df.columns:
    sns.distplot(df[feat])
    plt.xlabel(feat)
    plt.show()


# In[43]:


df


# # Train-Test

# In[44]:


X=df.drop(['overall_rating'],axis=1)
y=df['overall_rating']


# In[45]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=77)


# In[46]:


from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
X_train_scaled=scale.fit_transform(X_train)
X_test_scaled=scale.transform(X_test)


# In[47]:


y_train.shape


# In[48]:


import pickle
pickle.dump(scale,open('ml_scale.pickle','wb'))


# In[49]:


y_train_scaled=scale.fit_transform(y_train.values.reshape(-1, 1))
y_test_scale=scale.transform(y_test.values.reshape(-1, 1))


# In[50]:



pickle.dump(scale,open('ml_scale_y.pickle','wb'))


# In[48]:


from sklearn.linear_model import LinearRegression
lin=LinearRegression()
lin.fit(X_train_scaled,y_train_scaled)


# In[49]:


y_test_scale.shape


# In[50]:


pred=lin.predict(X_test_scaled)


# In[55]:


lin.score(X_test_scaled,y_test_scale)


# In[52]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test_scale,pred)


# In[53]:


from sklearn.linear_model import RidgeCV,Ridge
alphas = np.random.uniform(low=0, high=10, size=(50,))
ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)
ridgecv.fit(X_train_scaled, y_train_scaled)


# In[54]:


ridge=Ridge(alpha=ridgecv.alpha_)
ridge.fit(X_train_scaled,y_train_scaled)


# In[56]:


ridge.score(X_test_scaled, y_test_scale)


# In[57]:


pred_ridge=ridge.predict(X_test_scaled)


# In[58]:


mean_squared_error(y_test_scale,pred_ridge)


# In[61]:


from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor()
forest.fit(X_train_scaled,np.ravel(y_train_scaled))


# In[62]:


pred_forest=forest.predict(X_test_scaled)


# In[63]:


forest.score(X_test_scaled,np.ravel(y_test_scale))


# In[64]:


mean_squared_error(np.ravel(y_test_scale),pred_forest)


# In[66]:


import pickle
pickle.dump(forest,open('ml_model.pickle','wb'))


# In[71]:


from sklearn.model_selection import GridSearchCV

param_grid = {  'bootstrap': [True], 'max_depth': [5,7, 10 ], 'max_features': ['auto', 'log2'], 'n_estimators': [50,75,150,]}


# In[72]:


random=RandomForestRegressor()
grid = GridSearchCV(estimator = random, param_grid = param_grid, 

                          cv = 3, n_jobs = 1, verbose = 5, return_train_score=True)


# In[73]:


grid.fit(X_train_scaled, np.ravel(y_train_scaled))


# In[74]:


print(grid.best_params_)


# In[76]:


random=RandomForestRegressor(bootstrap=True,max_depth=10,max_features='log2',n_estimators=150)


# In[77]:


random.fit(X_train_scaled, np.ravel(y_train_scaled))


# In[78]:


random.score(X_test_scaled,np.ravel(y_test_scale))


# In[ ]:




