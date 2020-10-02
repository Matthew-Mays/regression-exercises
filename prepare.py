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



def prep_mall_data(df):
    '''
    Takes the acquired mall data, does data prep, and returns
    train, validate, and test data splits
    '''
    df['is_female'] = (df.gender == 'Female').astype('int')
    train_validate, test = train_test_split(df, test_size=.15, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.15, random_state=123)
    return train, validate, test


def prep_telco_data(df):
    # Drop duplicates if there are any
    df.drop_duplicates(inplace=True)
    #
    payment_type_cols = [1, 2, 3, 4]
    df['is_automatic_payment'] = df.payment_type_id.replace(payment_type_cols, [0,0,1,1])
    no_internet = ['streaming_movies', 'tech_support', 'streaming_tv', 'online_security', 'online_backup', 'device_protection']
    df[no_internet] = df[no_internet].replace({'No internet service': 0, 'No': 1, 'Yes': 2})
    binary_cols = ['churn', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'paperless_billing']
    df[binary_cols] = df[binary_cols].replace('Yes', 1).replace('No', 0)
    df.rename(columns={'gender':'male'}, inplace=True)
    df['male'] = df['male'].replace('Male', 1).replace('Female', 0)
    df['total_charges'] = df.total_charges.where((df.tenure != 0), 0)
    df = df.astype({'total_charges':'float64'})
    df['phone_lines'] = df['multiple_lines'].replace({'No phone service': 0, 'No': 1, 'Yes': 2})
    df.drop(columns='multiple_lines')
    df.rename(columns={'tenure':'monthly_tenure'}, inplace=True)
    df['yearly_tenure'] = round(df.monthly_tenure / 12, 2)
    df['part_depend'] = df['partner'] + df['dependents']
    df = df.drop(df[((df['monthly_tenure'].sort_values() == 0))].index)
    internet_types = [1, 2, 3]
    df['fiber'] = df['internet_service_type_id'].replace(internet_types, [0, 1, 0])
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)
    return train, validate, test

def add_scaled_columns(train, validate, test, scaler, columns_to_scale):
    new_column_names = [c + '_scaled' for c in columns_to_scale]
    scaler.fit(train[columns_to_scale])
    train = pd.concat([
        train,
        pd.DataFrame(scaler.transform(train[columns_to_scale]), columns=new_column_names, index=train.index),
    ], axis=1)
    
    validate = pd.concat([
        validate,
        pd.DataFrame(scaler.transform(validate[columns_to_scale]), columns=new_column_names, index=validate.index)
    ], axis=1)
    
    test = pd.concat([
        test,
        pd.DataFrame(scaler.transform(test[columns_to_scale]), columns=new_column_names, index=test.index)
    ], axis=1)
    
    return train, validate, test