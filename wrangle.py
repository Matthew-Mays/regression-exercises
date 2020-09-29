#!/usr/bin/env python
# coding: utf-8

# # Wrangle Exercises

# In[31]:


import acquire
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Throughout the exercises for Regression in Python lessons, you will use the following example scenario: As a customer analyst, I want to know who has spent the most money with us over their lifetime. I have monthly charges and tenure, so I think I will be able to use those two attributes as features to estimate total_charges. I need to do this within an average of $5.00 per customer.
# 
# The first step will be to acquire and prep the data. Do your work for this exercise in a file named wrangle.py.

# 1. Acquire customer_id, monthly_charges, tenure, and total_charges from telco_churn database for all customers with a 2 year contract.

# In[54]:


df = acquire.get_telco_data()
df = df[df['contract_type_id'] == 3]
df = df[['customer_id', 'monthly_charges', 'tenure', 'total_charges']]
df.head()


# 2. Walk through the steps above using your new dataframe. You may handle the missing values however you feel is appropriate.

# In[55]:


df.shape


# In[56]:


df.dtypes


# In[57]:


df.describe()


# In[58]:


df = df[df['tenure'] != 0]
df['total_charges'] = df['total_charges'].astype('float')
df.dtypes


# In[59]:


df.describe()


# In[1]:


# for col in ['monthly_charges', 'tenure', 'total_charges']:
#     plt.hist(df[col])
#     plt.title(col)
#     plt.show()


# In[2]:


#sns.boxplot(data=df[['monthly_charges']])


# In[3]:


#sns.boxplot(data=df[['tenure']])


# In[4]:


#sns.boxplot(data=df[['total_charges']])


# In[44]:


df.isna().sum()


# 3. End with a python file wrangle.py that contains the function, wrangle_telco(), that will acquire the data and return a dataframe cleaned with no missing values.

# In[65]:


def wrangle_telco(df):
    df = df[df['contract_type_id'] == 3]
    df = df[['customer_id', 'monthly_charges', 'tenure', 'total_charges']]
    df = df[df['tenure'] != 0]
    df['total_charges'] = df['total_charges'].astype('float')
    return df


# In[66]:


df = acquire.get_telco_data()
df.head()


# In[67]:


df = wrangle_telco(df)
df


# In[5]:


def wrangle_grades():
    grades = pd.read_csv("student_grades.csv")
    grades.drop(columns="student_id", inplace=True)
    grades.replace(r"^\s*$", np.nan, regex=True, inplace=True)
    df = grades.dropna().astype("int")
    return df


# In[ ]:




