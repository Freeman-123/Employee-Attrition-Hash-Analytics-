#!/usr/bin/env python
# coding: utf-8

# In[11]:


#K Means Cluster Analysis
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Attrition.csv")
print (df)
X = df.iloc[:, [1,2]].values #Selecting independent variables
print(X)


# In[12]:


#fitting k-means to dataset
from sklearn.cluster import KMeans
kms = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0)
Ykms = kms.fit_predict(X)
print(Ykms)


# In[14]:


plt.scatter(X[Ykms==0,0],X[Ykms==0,1],s=100,color='blue')
plt.scatter(X[Ykms==1,0],X[Ykms==1,1],s=100,color='grey')
plt.scatter(X[Ykms==2,0],X[Ykms==2,1],s=100,color='magenta')
plt.scatter(kms.cluster_centers_[:,0], kms.cluster_centers_[:,1], s=300, c='yellow', label='centroid')
plt.title(' 3 Clusters of Existing Employees')
plt.xlabel('Satisfaction Level')
plt.ylabel('Last Evaluation')
plt.savefig('cluster.png', dpi=500)
plt.legend()
plt.show()


# In[ ]:




