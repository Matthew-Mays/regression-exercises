#!/usr/bin/env python
# coding: utf-8

# ### The end product of this exercise should be the specified functions in a python script named prepare.py. Do these in your classification_exercises.ipynb first, then transfer to the prepare.py file.
# 
# ### Using the Iris Data:
# 
# 1. Use the function defined in acquire.py to load the iris data.

# In[75]:


import acquire
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

import warnings
warnings.filterwarnings('ignore')

import numpy as np


# 2. Drop the species_id and measurement_id columns.

# 3. Rename the species_name column to just species.

# 4. Create dummy variables of the species name.

# 5. Create a function named prep_iris that accepts the untransformed iris data, and returns the data with the transformations above applied.

# In[76]:


def prep_iris():
    df = acquire.get_iris_data()
    df = df.drop(columns=['species_id'])
    df = df.rename(columns={'species_name': 'species'})
    species_dummies = pd.get_dummies(df.species)
    df = pd.concat([df, species_dummies], axis=1)
    
    return df


# In[77]:


prep_iris()


# In[78]:


def titanic_split(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.survived)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.survived)
    return train, validate, test


# In[79]:


def impute_mean_age(train, validate, test):
    imputer = SimpleImputer(strategy = 'mean')
    train['age'] = imputer.fit_transform(train[['age']])
    validate['age'] = imputer.fit_transform(validate[['age']])
    test['age'] = imputer.fit_transform(test[['age']])
    
    return train, validate, test


# In[87]:


def prep_titanic():
    
    df = acquire.get_titanic_data()
    df = df[~df.embarked.isnull()]
    titanic_dummies = pd.get_dummies(df[['embarked', 'sex']], drop_first=True)
    df = pd.concat([df, titanic_dummies], axis=1)
    df = df.drop(columns=['deck', 'passenger_id', 'sex', 'embarked', 'embark_town', 'class'])
    train, validate, test = titanic_split(df)
    train, validate, test = impute_mean_age(train, validate, test)
    
    return train, validate, test


# In[88]:


train, validate, test = prep_titanic()
validate


# In[ ]:




