#!/usr/bin/env python
# coding: utf-8

# Implement a KNN model to classify the animals in to categorie
# 

# In[23]:


#Importing liabrary
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
df=pd.read_csv(r"C:\Users\anupa\Downloads\Zoo.csv")


# In[24]:


df.head()


# In[25]:


df.shape


# In[26]:


df.duplicated().sum()


# In[27]:


df.info()


# In[28]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[29]:


sns.countplot(data=df, x="type")


# # Train test split

# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


x=df.iloc[:,1:17]
y=df['type']


# In[32]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


# # KNN (K Neighrest Neighbour Classifier)

# In[33]:


from sklearn.model_selection import cross_val_score


# In[35]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=2)


# In[36]:


k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    train_scores = cross_val_score(knn, x_train, y_train, cv=5)
    k_scores.append(train_scores.mean())
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# In[37]:


model.fit(x_train, y_train)


# In[38]:


model.score(x_test, y_test)


# # Plot Confusion Matrix

# In[39]:


from sklearn.metrics import confusion_matrix
pred = model.predict(x_test)
cm = confusion_matrix(y_test, pred)
cm


# In[40]:


pred_df = pd.DataFrame({'Actual' : y_test, 'Predicted' : pred})
pred_df.head()


# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.figure(figsize=(7,5))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # Grid Search for Algorithm Tuning

# In[42]:


# Grid Search for Algorithm Tuning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


# In[43]:


n_neighbors = np.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors)


# In[44]:


model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(x_train, y_train)


# In[45]:


print(grid.best_score_)
print(grid.best_params_)


# In[ ]:




