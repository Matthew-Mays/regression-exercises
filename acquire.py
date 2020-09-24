#!/usr/bin/env python
# coding: utf-8

# In[2]:


import env
import pandas as pd
import os


# In[13]:


# Make a function named get_titanic_data that returns the titanic data from the codeup data science 
# database as a pandas data frame. Obtain your data from the Codeup Data Science Database. 

def get_connection(db, user=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
def get_titanic_data():
    filename = 'titanic.csv'
    if os.path.isfile(filename) == False:
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))
        df.to_csv(filename)
    else:
        df = pd.read_csv('titanic.csv', index_col=0)
    return df


# In[14]:


# Make a function named get_iris_data that returns the data from the iris_db on the codeup data science database 
# as a pandas data frame. The returned data frame should include the actual name of the species in addition 
# to the species_ids. Obtain your data from the Codeup Data Science Database.

def get_iris_data():
    filename = 'iris.csv'
    sql_query = '''SELECT species_id, species_name, sepal_length, sepal_width, petal_length, petal_width FROM
                measurements JOIN species USING(species_id)'''
    if os.path.isfile(filename) == False:
        df = pd.read_sql(sql_query, get_connection('iris_db'))
    else:
        df = pd.read_csv(filename)
        df.to_csv(filename)
    return df


# In[5]:


# Once you've got your get_titanic_data and get_iris_data functions written, now it's time to add caching to them. 
# To do this, edit the beginning of the function to check for a local filename like titanic.csv or iris.csv. 
# If they exist, use the .csv file. If the file doesn't exist, then produce the SQL and pandas necessary 
# to create a dataframe, then write the dataframe to a .csv file with the appropriate name.


#################################    Acquire Mall Customers Data     ################################################

def new_mall_data():
    '''
    This function reads the mall data from the Codeup db 
    into a df, write it to a csv and returns df.
    '''
    sql_query = 'SELECT * FROM customers'
    df = pd.read_sql(sql_query, get_connection('mall_customers'))
    df.to_csv('mall_customers_df.csv')
    return df

def get_mall_data(cached=False):
    '''
    This function reads in mall customer data from Codeup db if cached == False
    or if cached == True reads in mall customer df from a csv file, returns df
    '''
    if cached or os.path.isfile('mall_customers_df.csv') == False:
        df = new_mall_data()
    else:
        df = pd.read_csv('mall_customers_df.csv', index_col=0)
    return df


def get_telco_data():
    filename = 'telco.csv'
    if os.path.isfile(filename) == False:
        df = pd.read_sql('''SELECT * FROM customers JOIN internet_service_types 
                        USING(internet_service_type_id)
                        JOIN contract_types USING(contract_type_id)
                        JOIN payment_types USING(payment_type_id)''', get_connection('telco_churn'))
        df.to_csv(filename)
    else:
        df = pd.read_csv(filename, index_col=0)
    return df