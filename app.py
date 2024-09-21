#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# In[19]:


# Read original dataset
iris_df = pd.read_csv(r"iris.csv")
iris_df.sample(frac=1, random_state=40)
# selecting features and target data
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df[['species']]
# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=100)
# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)
# train the classifier on the training data
clf.fit(X_train, y_train)
# predict on the test set
y_pred = clf.predict(X_test)
# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}') # Accuracy: 0.91


# In[20]:


# save the model to disk
joblib.dump(clf, "rf_model.sav")


# In[21]:


import streamlit as st 
import pandas as pd
import numpy as np
from prediction import predict


# In[22]:


st.title('Classifying Iris Flowers')
st.markdown('Toy model to play to classify iris flowers into \
setosa, versicolor, virginica')


# In[23]:


st.header('Plant Features')
col1, col2 = st.columns(2)
with col1:
    
    st.text('Sepal characteristics')
    sepal_l = st.slider('Sepal lenght (cm)', 1.0, 8.0, 0.5)
    sepal_w = st.slider('Sepal width (cm)', 2.0, 4.4, 0.5)
with col2:
    st.text('Pepal characteristics')
    petal_l = st.slider('Petal lenght (cm)', 1.0, 7.0, 0.5)
    petal_w = st.slider('Petal width (cm)', 0.1, 2.5, 0.5)


# In[24]:


#st.button("Predict type of Iris")


# In[25]:


if st.button("Predict type of Iris"):
    result = predict(np.array([[sepal_l, sepal_w, petal_l, petal_w]]))
    st.text(result[0])


# In[ ]:




