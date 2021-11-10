#!/usr/bin/env python
# coding: utf-8

# In[1]:


###Part1 Data Processing###


# In[2]:


import pandas as pd
import numpy as np
df = pd.read_csv("crx.data", sep = ",", header=None)


# In[3]:


new_names =  {0:'A1', 1:'A2', 2 :'A3', 3:'A4', 4:'A5', 5:'A6', 6:'A7', 7:'A8', 8:'A9', 9:'A10', 10:'A11', 11:'A12', 12:'A13', 13:'A14', 14:'A15', 15:'A16'}
df.rename(columns = new_names, inplace=True)
df.head() #To look at renamed dataset


# In[4]:


# While searching throug the data set, it was found that unknown values are represented by '?'
#df.head(df.shape[0])


# In[5]:


## Subsequent steps remove the rows with unknown values
df.replace("?", np.nan, inplace = True)
print("Missing data")
print(df.isna().sum()) # to check whether it is the same as in description crx.names
num_samples = df.shape[0]
df.dropna(inplace=True)
print("Number of removed samples:", (num_samples - df.shape[0]))


# In[6]:


## Inspecting and converting data types
print(df.info()) # inspect
df['A2'] = df['A2'].astype(float)
df['A14'] = df['A14'].astype(int)
print(df.dtypes) # see the result of conversion


# In[7]:


## Encoding of categorical data


# In[8]:


#for true/false values the first dummy variable can be dropped (two values are not needed to represeny true/false)
df = pd.get_dummies(df, columns = ['A4', 'A5','A6','A7','A13'])
df = pd.get_dummies(df, columns = ['A1', 'A9', 'A10', 'A12', 'A16'], drop_first=True)
print(df.dtypes)
#df.head()


# In[9]:


# Splitting data into input(X) and output(y)
X = df.loc[:, "A2":"A12_t"]
y = df["A16_-"]


# In[10]:


# Splitting data into training and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y)


# In[11]:


# Scaling the data by standardization
from sklearn.preprocessing import StandardScaler
std_scaler = StandardScaler()
std_scaler.fit(X_train)
X_train_std = std_scaler.transform(X_train)

# Scaling the data by nminmaxscaler
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler()
mm_scaler.fit(X_train)
X_train_mm = mm_scaler.transform(X_train)


# In[12]:


###Part2 Training###


# In[13]:


from sklearn.linear_model import LogisticRegression

# a) class_weight none by default
logreg_a = LogisticRegression(penalty = 'none', max_iter = 200)
logreg_a.fit(X_train_std, y_train)


# In[14]:


# c) class_weight = balanced
logreg_c = LogisticRegression(penalty = 'none', class_weight = 'balanced', max_iter = 200)
logreg_c.fit(X_train_std, y_train)


# In[15]:


### Part3 Evaluation###


# In[16]:


# Scaling X test data according to the scaler that was used on training data
X_test_std = std_scaler.transform(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score


# In[17]:


## Prediction with minmaxscalar to compare performance
X_test_mm = mm_scaler.transform(X_test)
logreg_mm = LogisticRegression(penalty = 'none', max_iter = 250)
logreg_mm.fit(X_train_mm, y_train)
y_test_predict = logreg_mm.predict(X_test_mm)
y_test_predict_dec = logreg_mm.decision_function(X_test_mm)

print("Results for class_weight parameter None")
print("Accuracy score is: ", accuracy_score(y_test,y_test_predict))
print("Balanced accuracy score is: ", balanced_accuracy_score(y_test,y_test_predict))
# A16 was encoded into A16_- which shows 0 when output is + and 1 when output is -
print("Confusion matrix is: ", confusion_matrix(y_test,y_test_predict))
plot_precision_recall_curve(logreg_mm, X_test_mm, y_test)
print("Average Precision: ", average_precision_score(y_test, y_test_predict_dec))


# In[18]:


## Class weight none
y_test_predict = logreg_a.predict(X_test_std)
y_test_predict_dec = logreg_a.decision_function(X_test_std)

print("Results for class_weight parameter None")
print("Accuracy score is: ", accuracy_score(y_test,y_test_predict))
print("Balanced accuracy score is: ", balanced_accuracy_score(y_test,y_test_predict))
# A16 was encoded into A16_- which shows 0 when output is + and 1 when output is -
print("Confusion matrix is: ", confusion_matrix(y_test,y_test_predict))
plot_precision_recall_curve(logreg_a, X_test_std, y_test)
print("Average Precision: ", average_precision_score(y_test, y_test_predict_dec))


# In[19]:


## Class weight balanced
y_test_predict_balanced = logreg_c.predict(X_test_std)
y_test_predict_balanced_dec = logreg_c.decision_function(X_test_std)

print("Results for class_weight parameter None")
print("Accuracy score is: ", accuracy_score(y_test,y_test_predict_balanced))
print("Balanced accuracy score is: ", balanced_accuracy_score(y_test,y_test_predict_balanced))
# A16 was encoded into A16_- which shows 0 when output is + and 1 when output is -
print("Confusion matrix is: ", confusion_matrix(y_test,y_test_predict_balanced))
plot_precision_recall_curve(logreg_a, X_test_std, y_test)
print("Average Precision: ", average_precision_score(y_test, y_test_predict_balanced_dec))


# In[20]:


###Part 4###


# In[21]:


# Logistic Regression with penalty = 'l2'
logreg_regularized = LogisticRegression(penalty = 'l2')
logreg_regularized.fit(X_train_std, y_train)
y_test_predict_regularized = logreg_regularized.predict(X_test_std)
y_test_predict_regularized_dec = logreg_regularized.decision_function(X_test_std)

print("Results with regularization")
print("Accuracy score is: ", accuracy_score(y_test,y_test_predict_regularized))
print("Balanced accuracy score is: ", balanced_accuracy_score(y_test,y_test_predict_regularized))
# A16 was encoded into A16_- which shows 0 when output is + and 1 when output is -
print("Confusion matrix is: ", confusion_matrix(y_test,y_test_predict_regularized))
plot_precision_recall_curve(logreg_a, X_test_std, y_test)
print("Average Precision: ", average_precision_score(y_test, y_test_predict_regularized_dec))


# In[22]:


# Polynomial expansion attempt
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
print('Shape before: ', X_train.shape)
X_train_poly = poly.fit_transform(X_train)
print('Shape after polynomial expansion: ', X_train_poly.shape)

#standardization - it performs better when standardized
std_poly_scaler = StandardScaler()
std_poly_scaler.fit(X_train_poly)
X_train_poly_std = std_poly_scaler.transform(X_train_poly)

logreg_poly = LogisticRegression(solver = 'saga',max_iter = 10000)
logreg_poly.fit(X_train_poly_std, y_train)

y_test_predict_poly = logreg_poly.predict(poly.transform(X_test))
print("Accuracy score is: ", accuracy_score(y_test,y_test_predict_poly))

