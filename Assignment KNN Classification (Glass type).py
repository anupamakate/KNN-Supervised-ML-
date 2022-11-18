#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Prepare a model for glass classification using KNN

# Data Description:

# RI : refractive index

# Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)

# Mg: Magnesium

# AI: Aluminum

# Si: Silicon

# K:Potassium

# Ca: Calcium

# Ba: Barium

# Fe: Iron

# Type: Type of glass: (class attribute)
# 1 -- building_windows_float_processed
#  2 --building_windows_non_float_processed
#  3 --vehicle_windows_float_processed
#  4 --vehicle_windows_non_float_processed (none in this database)
#  5 --containers
#  6 --tableware
#  7 --headlamps
# # 


# In[2]:


import pandas as pd
import numpy as np

from seaborn import heatmap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score


# In[3]:


df_glass = pd.read_csv(r"C:\Users\anupa\Downloads\glass.csv")
df_glass


# In[4]:


df_glass.info()


# In[5]:


df_glass.Type.value_counts()


# In[6]:


corr = df_glass.corr()
corr


# In[7]:


heatmap(corr)


# features Ca and K has almost zero correlation with class type

# In[8]:


scaler = StandardScaler()
scaler.fit(df_glass.drop('Type', axis=1))

# Perform transformation

scaler_feat = scaler.transform(df_glass.drop('Type', axis=1))
scaler_feat


# In[9]:


df_glass_std = pd.DataFrame(scaler_feat, columns=df_glass.columns[:-1])
df_glass_std


# # KNN Model

# In[10]:


# Remove features Ca and K

X = df_glass_std.drop(['Ca', 'K'], axis=1)
X.head()


# In[11]:


Y = df_glass['Type']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=40)


# In[12]:


num_folds = 10
kfold = KFold(n_splits=10)


# In[13]:


model = KNeighborsClassifier(n_neighbors=17)
results = cross_val_score(model, x_train, y_train, cv=kfold)

print(results.mean())


# # Grid Search for Algorithm Tuning

# In[14]:


# Grid Search for Algorithm Tuning

n_neighbors = np.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors)

model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X, Y)


# In[15]:


print(grid.best_score_)
print(grid.best_params_)


# In[16]:


# Visualizing the CV results

import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


# Choose K between 1 to 41

k_range = range(1, 41)
k_scores = []

# Use iteration to caclulator different K in models, then return the average accuracy based on the cross validation

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=5)
    k_scores.append(scores.mean())
    
# Plot to see clearly

plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()


# ## For K = 3, best value, final model

# In[18]:


model = KNeighborsClassifier(n_neighbors=3)
results = cross_val_score(model, x_train, y_train, cv=kfold)

print(results.mean())


# In[19]:


model.fit(x_train, y_train)


# In[20]:


y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))


# In[21]:


accuracy_score(y_test, y_pred)


# In[ ]:





# In[ ]:




