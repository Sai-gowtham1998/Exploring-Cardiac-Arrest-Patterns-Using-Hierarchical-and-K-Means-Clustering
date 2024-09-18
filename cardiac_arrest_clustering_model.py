#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv(r"C:\Gowtham\Finger tips\All Projects\Python + ML\ML Project - Clustering Cardiac Arrest Prediction\Cardiac_diagnostic_features.csv")


# In[4]:


df.head()


# In[6]:


df.shape


# In[5]:


df.info()


# # Checking for null values 

# In[7]:


df.isnull().sum()


# In[8]:


df.drop(["Unnamed: 32", "id"], axis = 1, inplace = True)
df.head()


# In[8]:


df["diagnosis"].value_counts()


# In[9]:


df1 = df.drop(["diagnosis"], axis = 1)
df1.head()


# # Create pair plot between two column radius_mean and radius_mean by diagnosis

# In[10]:


sns.pairplot(df.loc[:,['radius_mean','texture_mean', 'diagnosis']], hue = "diagnosis", height = 5)
plt.show()


# # Select only two feature radius_mean & texture_mean for clustering in new dataset

# In[10]:


new_data = df.loc[:,['radius_mean','texture_mean']]
new_data.head()


# # Applying scaling on new dataset

# In[11]:


#perform standard scalar
from sklearn.preprocessing import StandardScaler


# In[12]:


sc = StandardScaler()

sc = sc.fit(new_data)


# In[13]:


data = sc.fit_transform(new_data)


# In[14]:


data =pd.DataFrame(data , columns=['radius_mean','texture_mean'])


# In[15]:


data.head()


# # Displaying hierarchical clustering as a dendrogram using scipy

# In[17]:


#The following linkage methods are used to compute the distance between two clusters 
# method='ward' uses the Ward variance minimization algorithm
from scipy.cluster.hierarchy import linkage,dendrogram
merg = linkage(data, method = "ward")
#Plot the hierarchical clustering as a dendrogram.
#leaf_rotation : double, optional Specifies the angle (in degrees) to rotate the leaf labels.
dendrogram(merg, leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()


# # Apply AgglomerativeClustering on dataset with 2 n number of cluster

# In[16]:


from sklearn.cluster import AgglomerativeClustering


# # Predict the cluseter and create new column for cluster lable data

# In[17]:


hc = AgglomerativeClustering(n_clusters = 2, affinity = "euclidean", linkage = "ward")
cluster = hc.fit_predict(data)


# In[18]:


data["label"] = cluster


# In[19]:


data.head()


# # Checking count of label 

# In[20]:


data.label.value_counts()


# # Plot the lable data 

# In[23]:


plt.figure(figsize = (15, 10))
plt.scatter(data["radius_mean"][data.label == 0], data["texture_mean"][data.label == 0], color = "red")
plt.scatter(data["radius_mean"][data.label == 1], data["texture_mean"][data.label == 1], color = "blue")

plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.show()


# # Check the  silhouette score 

# In[24]:


from sklearn.metrics import silhouette_score


# In[25]:


score_agg = silhouette_score(data, cluster)
score_agg


# In[ ]:





# # Now apply k-means clustering no dataset with 2 number of cluster

# In[21]:


new_data_k = df.loc[:,['radius_mean','texture_mean']]
new_data_k.head()


# In[22]:


from sklearn.cluster import KMeans


# In[29]:


cls = KMeans(n_clusters = 2, )
cls.fit(new_data_k)


# # Check wcss score

# In[30]:


wcss = cls.inertia_
wcss


# # Try differents N number from 1 to 10 and plot the result of wcss score 

# In[23]:


#create empty list
wcss = []
#select k value from 1 to 10
for i in range(1, 11):
    cls = KMeans(n_clusters = i, random_state = 42)
    cls.fit(new_data_k)
    # inertia method returns wcss for that model
    wcss.append(cls.inertia_)


# In[24]:


plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# # Apply k-means again with different no. of cluster according to  best wcss score 

# In[25]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(new_data_k)


# # Create column for label cluster 

# In[26]:


new_data_k["label"] = y_kmeans


# In[37]:


new_data_k.head()


# In[ ]:




