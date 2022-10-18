#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("Fraud.csv")


# In[3]:


df.head()


# In[4]:


df.shape


#                 DATA DICTIONARY
# 
# step - maps a unit of time in the real world. In this case 1 step is 1 hour of time. Total steps 744 (30 days simulation).
# 
# type - CASH-IN, CASH-OUT, DEBIT, PAYMENT and TRANSFER.
# 
# amount - amount of the transaction in local currency.
# 
# nameOrig - customer who started the transaction
# 
# oldbalanceOrg - initial balance before the transaction
# 
# newbalanceOrig - new balance after the transaction
# 
# nameDest - customer who is the recipient of the transaction
# 
# oldbalanceDest - initial balance recipient before the transaction. Note that there is not information for customers that start with M (Merchants).
# 
# newbalanceDest - new balance recipient after the transaction. Note that there is not information for customers that start with M (Merchants).
# 
# isFraud - This is the transactions made by the fraudulent agents inside the simulation. In this specific dataset the fraudulent behavior of the agents aims to profit by taking control or customers accounts and try to empty the funds by transferring to another account and then cashing out of the system.
# 
# isFlaggedFraud - The business model aims to control massive transfers from one account to another and flags illegal attempts. An illegal attempt in this dataset is an attempt to transfer more than 200.000 in a single transaction.

# In[5]:


df.describe()


# In[6]:


df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[9]:


df_new = df[df['nameDest'].str.contains('M') == False]
df_new.head()


# In[47]:


df_new.corr()


# In[67]:


total_transactions = len(df_new)
fraud_transactions = len(df_new[df_new.isFraud == 1])
print(fraud_transactions)
print(total_transactions)
print("Percent of fraud transactions is: {:.4f} %".format((fraud_transactions * 100)/ total_transactions))


# In[12]:


def correlation_heatmap(df):
    correlations = df.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show();
    
correlation_heatmap(df)


# From the above heatmap, it is clear that the feature 'oldbalanceOrg' has high correlation with 'newbalanceorg' and feature 'oldbalanceDest' has high correlation with 'newbalanceDest'.Therefore, to build a machine learning model with good accuracy, these columns need to be removed or operated, so that there is no skewer in the final output of the model.
# so, we can drop either of them.

# In[30]:


df["finalbalanceOrig"] = df['newbalanceOrig'] - df['oldbalanceOrg']
df["finalbalanceDest"] = df['newbalanceDest'] - df['oldbalanceDest']


# In[31]:


df_new = df.drop(columns=['oldbalanceOrg','newbalanceOrig', 'oldbalanceDest', 'newbalanceDest'])


# In[32]:


df_new.head()


# In[33]:


df_new['type'].unique()


# In[35]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()

# transforming the column after fitting
enc = enc.fit_transform(df_new[['type']]).toarray()

# converting arrays to a dataframe
encoded_colm = pd.DataFrame(enc)

# concating dataframes 
df_new = pd.concat([df_new, encoded_colm], axis = 1) 

# removing the encoded column.
df_new= df_new.drop(['type'], axis = 1) 
df_new.head(10)


# Displaying the heatmap for new dataset with onehot encoded data

# In[62]:


correlation_heatmap(df_new)


# MODEL DEVELOPMENT

# In[64]:


get_ipython().system('pip install xgboost')


# In[18]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import itertools
from collections import Counter



# In[ ]:


from sklearn.model_selection import train_test_split as split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
data_dummy = pd.get_dummies(data)
data_dummy.columns = data_dummy.columns.str.replace(' ','_')
train, test = split(data_dummy, test_size = .30, random_state = 12)
train.shape
train.head(2)
16
X_train = train.drop('target', axis = 1)
Y_train = train.target
X_test = test.drop('target', axis = 1)
Y_test = test.target
lr = LogisticRegression()
lr.fit(X_train,Y_train)
pred = lr.predict(X_test)
accuracy_score(y_true = Y_test,y_pred = pred)
print(classification_report(y_true=Y_test,y_pred = pred))


# In[71]:


df_new.head()


# In[60]:


import pickle


# In[61]:


filename = 'xgboost_model.pickle'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))


# CONCLUSION

# 1. Data cleaning including missing values, outliers and multi-collinearity.
# so in the given data set there is no missing value,
# the multicollinearity between the columns and find the decisive factors for 
# fraud detection in the account transactions was checked.
# 
# 2. Describe your fraud detection model in elaboration.
#  After cleaning the data and performing data analysis on the it, the produced dataset is used in training of the machine learning model.
# In this task, I have used Logistic Regression test and XGBOOST classification to develop the model.
# 
# 
# 3. How did you select variables to be included in the model?
#  The list variables or features are - [step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud].
# Out of these features, 'step', 'nameOrig', and 'nameDest' are the features which have no role to play in prediction a trasaction being fraud or not fraud.
# This is because these are just the unique string values, which can be ignored.
# The main role for classifying a prediction as fraud or not fraud are played by the difference between the original and 
# new_balance amount in the accounts of sender and receiver, further depending upon the 'type of transaction'. 
# Therefore, the features considered for classification tasks are oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest and type.
# 
# 4. What are the key factors that predict fraudulent customer?
# A sequence of transactions is made by customer to one another in the same step
# The amount of transaction is equal to the oldbalance of the sender's account.
# 
# 6. Do these factors make sense? If yes, How? If not, How not?
# This is the transactions made by the fraudulent agents inside the simulation. 
# In this specific dataset the fraudulent behavior of the customers aims to profit by taking control or 
# customers accounts and try to empty the funds by transferring to another account and 
# then cashing out of the system is Fraud
# 
# 7. What kind of prevention should be adopted while company update its infrastructure?
# strong login password with captcha, control massive transfers from one account to another and flags illegal attempts. 
# An illegal attempt means transaction of more than 2 lakh in single attempts.
# 
# 8. Assuming these actions have been implemented, how would you determine if they work?
# customers keep checking there account activity,Company sending account details to customers by sms and emails.
# 

# 
