#!/usr/bin/env python
# coding: utf-8


# In[7]:


import pandas as pd
import numpy as np
zomato=pd.read_csv('zomato.csv')


# In[8]:


zomato.head()


# In[9]:


zomato.dtypes


# In[10]:


len(zomato)


# In[11]:


zomato.isnull().sum()/len(zomato)


# In[12]:


pd.set_option('display.max_rows', 500)
zomato['rate'].unique()


# In[13]:


zomato['rate'].dtype


# In[14]:


zomato[zomato['rate'].isnull()]


# In[15]:


test_zomato=zomato[zomato['rate'].isnull()]
test_zomato.head()


# In[16]:


zomato.drop(['url','phone','dish_liked'], axis =1, inplace=True)


# In[17]:


zomato.dropna(how='any',inplace=True)


# In[18]:


zomato.isnull().sum()


# # EDA****

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[20]:


cat_var=[feat for feat in zomato.columns if zomato[feat].dtype=='O' and feat!='rate']


# In[21]:


cat_rare_var=[feat for feat in cat_var if zomato[feat].nunique()<=25]


# In[22]:


zomato['rate']=zomato['rate'].astype(str)
zomato['rate']=zomato['rate'].apply(lambda x:x[:3])



# In[23]:


for feat in cat_rare_var:
    plt.figure(figsize=(15,15))
    plt.xticks(rotation=90)
    sns.countplot(data=zomato, x='rate', order=zomato.rate.value_counts().index,hue=feat)
    plt.show()


# In[24]:


plt.figure(figsize=(12,6))
sns.distplot(zomato['votes'])


# In[25]:


zomato['approx_cost(for two people)'] = zomato['approx_cost(for two people)'].astype(str)
zomato['approx_cost(for two people)'] = zomato['approx_cost(for two people)'].apply(lambda x: x.replace(',','.'))
zomato['approx_cost(for two people)'] = zomato['approx_cost(for two people)'].astype(float)
zomato.info()


# In[26]:


plt.figure(figsize=(20,7))
plt.xticks(rotation=90)
sns.countplot('location', data=zomato)


# In[27]:


plt.figure(figsize=(20,7))
sns.countplot(x='listed_in(type)',data=zomato)
plt.xticks(rotation=90)


# In[28]:


plt.figure(figsize=(15,7))
top=zomato['name'].value_counts()[:25]
sns.barplot(x=top,y=top.index,palette='Set1')
plt.title("25 Most Famous restaurant chains in Bangaluru on Zomato",pad=20)
plt.xlabel("Number of outlets")


# In[29]:


zomato['votes'].value_counts()


# In[30]:


plt.figure(figsize=(14,7))
x=zomato.groupby('name')['votes'].max().nlargest(20).index
y=zomato.groupby('name')['votes'].max().nlargest(20)
print(y)
sns.barplot(x,y,palette='Set1')
plt.title("20 Most Voted restaurant chains in Bangaluru on Zomato",pad=20)
plt.xticks(rotation=90)
plt.xlabel("Name of outlet")


# In[31]:


plt.figure(figsize=(20,7))
x=zomato.groupby('name')['approx_cost(for two people)'].max().nlargest(20).index
y=zomato.groupby('name')['approx_cost(for two people)'].max().nlargest(20)
sns.barplot(x,y,palette='Set1')
plt.title('Top 20 Most Expenseive Zomato chains',pad=20)
plt.xticks(rotation=90)
plt.xlabel('Name of outlet')


# In[32]:


plt.figure(figsize=(20,7))
x=zomato['location'].value_counts()[:20].index
y=zomato['location'].value_counts()[:20]
sns.barplot(x,y,palette='Set1')
plt.title('Top 20 Most Dense Location',pad=20)
plt.xticks(rotation=90)
plt.xlabel('Name of outlet')


# In[33]:


zomato['location'].value_counts()


# In[34]:


plt.figure(figsize=(20,7))
plt.subplot(3,1,1)
sns.scatterplot(x="rate",y='approx_cost(for two people)',hue='online_order',data=zomato)
plt.subplot(3,1,2)
sns.scatterplot(x="rate",y='online_order',hue='votes',data=zomato)
plt.subplot(3,1,3)
sns.barplot(x='votes',y='online_order',data=zomato)
plt.show()


# In[35]:


plt.subplot(1,2,1)
sns.boxplot(x='online_order',y='votes',data=zomato)


# In[36]:


zomato.isnull().sum()


# In[37]:


zomato['rate'].unique()


# # Preprocessing****

# In[38]:



zomato['rate'].value_counts()


# In[39]:


zomato.drop(['address'],axis=1,inplace=True)


# In[40]:


zomato=pd.concat([zomato,pd.get_dummies(zomato['online_order'],drop_first=True)],axis=1)
zomato.head()


# In[41]:


zomato.isnull().sum()


# In[42]:


zomato.drop(['online_order'],axis=1,inplace=True)


# In[43]:


df_yes=pd.get_dummies(zomato['book_table'],drop_first=True)


# In[44]:


df_yes


# In[45]:


df_yes.rename(columns={'Yes':'yes_book_table'},inplace=True)
df_yes


# In[46]:


zomato=pd.concat([zomato,df_yes],axis=1)


# In[47]:


zomato.drop(['book_table'],axis=1,inplace=True)
zomato.head()


# In[48]:


zomato['listed_in(type)'].value_counts()


# In[49]:


b=zomato.groupby('listed_in(city)')['votes'].count().sort_values(ascending=False).index


# In[50]:


l=[]
for i in zomato['rate']:
  if i not in ['NEW','-']:
    l.append(float(i))
med=np.median(np.array(l))


# In[51]:


zomato['rate'].replace(to_replace=['NEW','-'],value=med,inplace=True)


# In[52]:


zomato['rate']=zomato['rate'].astype(float)


# In[53]:


zomato['rate'].value_counts().sort_values()


# In[54]:


a=zomato.groupby('listed_in(city)')['rate'].mean().sort_values(ascending=False).index


# In[55]:


l=[]
for i in range(25):
  if a[i] in a[:25] :
    l.append(a[i])
l


# In[56]:


b=zomato.groupby('name')['votes'].count().sort_values(ascending=False).index
b


# In[57]:


zomato


# In[58]:



for i in l:
  zomato[i]=np.where(zomato['location']==i,1,0)


# In[59]:



for i in b[:25]:
  zomato[i]=np.where(zomato['name']==i,1,0)


# In[60]:


zomato.isnull().sum()


# In[61]:


zomato.drop(['reviews_list','cuisines','listed_in(city)'],axis=1,inplace=True)


# In[62]:


zomato.shape


# In[63]:


zomato.isnull().sum()


# In[64]:


zomato['votes'].unique()


# In[65]:


zomato.drop(['menu_item'],axis=1,inplace=True)


# In[66]:


zomato.isnull().sum()


# In[67]:


zomato=pd.concat([zomato.drop(['listed_in(type)'],axis=1),pd.get_dummies(zomato['listed_in(type)'],drop_first=True)],axis=1)


# In[68]:


zomato.isnull().sum()


# In[69]:


a=zomato.groupby('rest_type')['votes'].count().sort_values(ascending=False).index
a


# In[70]:


b=zomato.groupby('rest_type')['rate'].mean().sort_values(ascending=False).index
b


# In[71]:


l=[]
for i in range(10):
  if i in a[:25] and i in b[:25]:
    l.append(i)
l


# In[72]:



for i in a[:10]:
  zomato[i]=np.where(zomato['rest_type']==i,1,0)

  


# In[73]:


zomato.isnull().sum()


# In[74]:


zomato.head()


# In[75]:


zomato.drop(['name'],axis=1,inplace=True)


# In[76]:


zomato.drop(['location','rest_type'],axis=1,inplace=True)


# In[77]:


zomato.head()


# In[78]:


zomato.isnull().sum()


# In[79]:


plt.figure(figsize=(42,37))
sns.heatmap(zomato.corr(method = 'spearman'),annot=True)


# In[80]:


from scipy.stats import probplot
import pylab
for col in zomato.columns:
  plt.figure(figsize=(9,4))
  plt.subplot(1,2,1)
  zomato[col].hist()
  plt.xlabel(col)
  plt.subplot(1,2,2)
  probplot(zomato[col],dist='norm',plot=pylab)
plt.show()


# In[81]:


for col in zomato.columns:
  zomato[col]=np.sqrt(zomato[col])
for col in zomato.columns:
  plt.figure(figsize=(9,4))
  plt.subplot(1,2,1)
  zomato[col].hist()
  plt.xlabel(col)
  plt.subplot(1,2,2)
  probplot(zomato[col],dist='norm',plot=pylab)
plt.show()


# In[82]:


X=zomato.drop(['rate'],axis=1)


# In[83]:


y=zomato['rate']


# In[84]:


zomato.shape


# In[85]:


X.isnull().sum()


# In[86]:


from sklearn.feature_selection import SelectKBest, mutual_info_regression
select = SelectKBest(mutual_info_regression, k=35)
X_new=select.fit_transform(X, y)


# In[87]:


X_new.shape


# In[88]:


select.get_support(indices=False)


# In[89]:


from sklearn.preprocessing import MinMaxScaler


# In[90]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=22)


# In[91]:


scale=MinMaxScaler()
X_train_scaled=scale.fit_transform(X_train)


# In[92]:


X_train_scaled


# In[93]:


X_test_scaled=scale.transform(X_test)


# In[94]:


import pickle
pickle.dump(scale,open('min_max.pickle','wb'))


# In[95]:


y_train_scaled=scale.fit_transform(y_train.values.reshape(-1, 1))


# In[96]:


y_test_scaled=scale.transform(y_test.values.reshape(-1, 1))


# In[97]:


import pickle
pickle.dump(scale,open('min_max_y.pickle','wb'))


# In[98]:


from sklearn.cluster import KMeans
wcss=[]
for i in range (1,21):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(X_train_scaled)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,21),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[99]:


kmeans = KMeans(n_clusters = 6, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X_train_scaled)
print(y_kmeans)


# In[100]:


y_test_kmeans=kmeans.predict(X_test_scaled)


# In[101]:


np.unique(y_kmeans)


# In[102]:


cluster=pd.DataFrame(data=y_kmeans,columns=["Cluster"])


# In[103]:


cluster_test=pd.DataFrame(data=y_test_kmeans,columns=["Cluster"])


# In[104]:


cluster.head()


# In[105]:


np.shape(X_train_scaled)[1]


# In[106]:


final_data=pd.DataFrame(data=X_train_scaled,columns=[i for i in range(np.shape(X_train_scaled)[1])])


# In[107]:


final_test=pd.DataFrame(data=X_test_scaled,columns=[i for i in range(np.shape(X_test_scaled)[1])])


# In[108]:


y_final=pd.DataFrame(data=y_train_scaled,columns=['target'])
y_final.head()


# In[109]:


y_test=pd.DataFrame(data=y_test_scaled,columns=['target'])
y_test.head()


# In[110]:


y_final=pd.concat([y_final,cluster],axis=1)


# In[111]:


y_test=pd.concat([y_test,cluster_test],axis=1)


# In[112]:


final_data


# In[113]:


final_data=pd.concat([final_data,cluster],axis=1)


# In[114]:


final_test=pd.concat([final_test,cluster_test],axis=1)


# In[115]:


data_0=final_data[final_data['Cluster']==0]
data_0


# In[116]:


data_y_0=y_final[y_final['Cluster']==0]
data_y_0


# In[117]:


data_1=final_data[final_data['Cluster']==1]
data_1


# In[118]:


data_y_1=y_final[y_final['Cluster']==1]
data_y_1


# In[119]:


data_2=final_data[final_data['Cluster']==2]
data_2.head()


# In[120]:


data_y_2=y_final[y_final['Cluster']==2]
data_y_2


# In[121]:


data_3=final_data[final_data['Cluster']==3]
data_3.head()


# In[122]:


data_y_3=y_final[y_final['Cluster']==3]
data_y_3


# In[123]:


data_4=final_data[final_data['Cluster']==4]
data_4.head()


# In[124]:


data_y_4=y_final[y_final['Cluster']==4]
data_y_4


# In[125]:


data_5=final_data[final_data['Cluster']==5]
data_5.head()


# In[126]:


data_y_5=y_final[y_final['Cluster']==5]
data_y_5


# In[127]:


from sklearn.linear_model import LinearRegression,Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost


# In[128]:


lin_0=LinearRegression()
lin_0.fit(data_0.drop(['Cluster'],axis=1),data_y_0.drop(['Cluster'],axis=1))


# In[129]:


pred_0=lin_0.predict(final_test[final_test['Cluster']==0].drop(['Cluster'],axis=1))


# In[130]:


y_final['Cluster'].unique()


# In[131]:


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test[y_test['Cluster']==0].drop(['Cluster'],axis=1),pred_0)


# In[132]:


pred_0


# In[133]:


y_test[y_test['Cluster']==0].drop(['Cluster'],axis=1)


# In[134]:


ridge_0=Ridge(alpha=2.0)
ridge_0.fit(data_0.drop(['Cluster'],axis=1),data_y_0.drop(['Cluster'],axis=1))


# In[135]:


pred_r_0=ridge_0.predict(final_test[final_test['Cluster']==0].drop(['Cluster'],axis=1))


# In[136]:


mean_squared_error(y_test[y_test['Cluster']==0].drop(['Cluster'],axis=1),pred_r_0)


# In[137]:


ridge_0.score(final_test[final_test['Cluster']==0].drop(['Cluster'],axis=1),np.ravel(y_test[y_test['Cluster']==0].drop(['Cluster'],axis=1),order='C'))


# In[138]:


from sklearn.ensemble import RandomForestRegressor


# In[139]:


random_0=RandomForestRegressor()
random_0.fit(data_0.drop(['Cluster'],axis=1),np.ravel(data_y_0.drop(['Cluster'],axis=1),order='C'))


# In[140]:


pred_forest_0=random_0.predict(final_test[final_test['Cluster']==0].drop(['Cluster'],axis=1))
mean_squared_error(y_test[y_test['Cluster']==0].drop(['Cluster'],axis=1),pred_forest_0)


# In[141]:


pred_forest_0


# In[142]:


random_0.score(final_test[final_test['Cluster']==0].drop(['Cluster'],axis=1),np.ravel(y_test[y_test['Cluster']==0].drop(['Cluster'],axis=1),order='C'))


# In[143]:


lin_1=LinearRegression()
lin_1.fit(data_1.drop(['Cluster'],axis=1),y_final[y_final['Cluster']==1].drop(['Cluster'],axis=1))


# In[144]:


pred_1=lin_1.predict(final_test[final_test['Cluster']==1].drop(['Cluster'],axis=1))
mean_squared_error(y_test[y_test['Cluster']==1].drop(['Cluster'],axis=1),pred_1)


# In[145]:


random_1=RandomForestRegressor()
random_1.fit(data_1.drop(['Cluster'],axis=1),np.ravel(data_y_1.drop(['Cluster'],axis=1),order='C'))


# In[146]:


pred_forest_1=random_1.predict(final_test[final_test['Cluster']==1].drop(['Cluster'],axis=1))
mean_squared_error(y_test[y_test['Cluster']==1].drop(['Cluster'],axis=1),pred_forest_1)


# In[147]:


pred_forest_1


# In[148]:


random_2=RandomForestRegressor()
random_2.fit(data_2.drop(['Cluster'],axis=1),np.ravel(data_y_2.drop(['Cluster'],axis=1),order='C'))


# In[149]:


pred_forest_2=random_2.predict(final_test[final_test['Cluster']==2].drop(['Cluster'],axis=1))
mean_squared_error(y_test[y_test['Cluster']==2].drop(['Cluster'],axis=1),pred_forest_2)


# In[150]:


random_3=RandomForestRegressor()
random_3.fit(data_3.drop(['Cluster'],axis=1),np.ravel(data_y_3.drop(['Cluster'],axis=1),order='C'))


# In[151]:


pred_forest_3=random_3.predict(final_test[final_test['Cluster']==3].drop(['Cluster'],axis=1))
mean_squared_error(y_test[y_test['Cluster']==3].drop(['Cluster'],axis=1),pred_forest_3)


# In[299]:


random_4=RandomForestRegressor(bootstrap= True,
 max_depth= 110,
 max_features= 'auto',
 max_leaf_nodes= 1450,
 min_samples_split= 2,
 n_estimators=160)
random_4.fit(data_4.drop(['Cluster'],axis=1),np.ravel(data_y_4.drop(['Cluster'],axis=1),order='C'))


# In[300]:


pred_forest_4=random_4.predict(final_test[final_test['Cluster']==4].drop(['Cluster'],axis=1))
mean_squared_error(y_test[y_test['Cluster']==4].drop(['Cluster'],axis=1),pred_forest_4)


# In[269]:


random_5=RandomForestRegressor(bootstrap= True,
 max_depth= 90,
 max_features= 'auto',
 max_leaf_nodes= 1350,
 min_samples_split= 2,
 n_estimators=150)
random_5.fit(data_5.drop(['Cluster'],axis=1),np.ravel(data_y_5.drop(['Cluster'],axis=1),order='C'))


# In[270]:


pred_forest_5=random_5.predict(final_test[final_test['Cluster']==5].drop(['Cluster'],axis=1))
mean_squared_error(y_test[y_test['Cluster']==5].drop(['Cluster'],axis=1),pred_forest_5)


# In[218]:


grid_forest=RandomForestRegressor()
param= { 
            "n_estimators"      : [130,150],
            "max_features"      : ["auto", "log2"],
            "min_samples_split" : [2,],
            "bootstrap": [True, False],
            "n_jobs" : [-1],
            "max_leaf_nodes" : [500,600],
            "max_depth" : [70,90],
            }
grid=GridSearchCV(grid_forest,param,cv=4)


# In[219]:



grid.fit(data_5.drop(['Cluster'],axis=1),np.ravel(data_y_5.drop(['Cluster'],axis=1),order='C'))


# In[220]:


pred_grid_4=grid.predict(final_test[final_test['Cluster']==4].drop(['Cluster'],axis=1))
mean_squared_error(y_test[y_test['Cluster']==4].drop(['Cluster'],axis=1),pred_grid_4)


# In[221]:


grid.best_params_


# In[ ]:




