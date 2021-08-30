#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds_E = pd.read_excel('Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx', sheet_name='Existing employees')
ds_E.head()
ds_L = pd.read_excel('Hash-Analytic-Python-Analytics-Problem-case-study-1.xlsx', sheet_name='Employees who have left')
ds_L.head()


# In[40]:


ds=pd.concat([ds_E, ds_L])
ds.info()


# In[42]:


ds_Target=ds['Emp_Status']
ds_Target.head()
ds_feature=ds.drop(['Emp_Status','dept','salary'], axis=1)
ds_feature.tail()


# In[43]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
Y = le.fit_transform(ds_Target)
print(Y)


# In[44]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(ds_feature,Y,test_size=0.3,random_state=42)
print (X_train)
print (X_test)
print (Y_test)
print (Y_train)


# In[45]:


from sklearn.tree import DecisionTreeClassifier
classifer = DecisionTreeClassifier()
classifer.fit(X_train,Y_train)


# In[46]:


pred = classifer.predict(X_test)
print(pred)


# In[47]:


from sklearn.metrics import accuracy_score
check= accuracy_score(Y_test,pred)
print(check)


# In[48]:


Y_pred = classifer.predict(ds_feature)
print(Y_pred)


# In[49]:


df=pd.DataFrame(data={"Emp ID":ds_feature["Emp ID"],"Emp Stay/Leave":Y_pred})

df.to_csv("Prone_to_leave.csv", index=False)


# In[50]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,pred)
print (cm)


# In[ ]:





# In[ ]:




