import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

Created in Visual Studio Code | Jupyter Notebook | Python Interpreter 3.12.2

file_path1 = 'C:/Users/navji/Downloads/NavjitSinghInterviewCode/people.csv'
file_path2 = 'C:/Users/navji/Downloads/NavjitSinghInterviewCode/services.csv'
df1 = pd.read_csv(file_path1)
df2 = pd.read_csv(file_path2)




Here we merge both tables into a pandas dataframe based on the unique identifier(id). 
We do an inner join to keep unique records relevant and because they are available.
The inner join also allows us to have one person be recorded for multiple events instead of as a separate person

df = pd.merge(df1, df2, on='id', how='inner')
df

Dropping columns that are created when merging as well as the count column as it is a constant 1 throughout the entire df, redundant data.

df = df.drop(columns=(['Unnamed: 0_x', 'Unnamed: 0_y', 'count']), axis=1)






Here I developed the age column by replacing the naans and null values with the average of the column. I also checked for other naans present throughout the code

#na1 = df.isna().sum()
#na1
df['age'] = df['age'].fillna(df['age'].mean())
#df.info()
df






Here we utilize the dummies function to unpack the events column into binary columns which will be used for further data analysis
Then dropped the event column after all data was transformed

dummies = pd.get_dummies(df['event'])
dummies = dummies.astype(int)
df = pd.concat([df, dummies], axis=1)
df = df.drop(columns=['event'])
df






Here I was checking for naans as well as how the ages were distributed throughout the dataframe, it was pretty evenly distributed and there were no outliers.
I did notice that we continued to use the same people for multiple events which increased our dataframe from less than 1000 rows to a few thousand

#maxage = df['age'].max()
#minage = df['age'].min()
#maxage
#minage







Understanding the formula of er_visit ~ predictor variables we must understand that the target variable is er_visit with other variables as the predictors
This makes the objective of this model a classification as er_visit is represented by a binary value not continuous.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from patsy import dmatrices
#Our objective is to have er_visit(classification) as target variable
#Use other events, and age as predictor variables


#formula = 'er_visit ~ age + date + address_change + civil_court_case + medicaid_enrollment + mental_health_service + physical_health_service + snap_enroll'
x = df.drop(['er_visit'], axis =1)
y = df['er_visit']







Here we split the data into training and testing, 
I decided to use 70/30 as there was not a lot of data and we should have more validation in case there were instances of certain variables 
lacking in the validation set with a lower amount(80/20)

#na1 = df.isna().sum()
#na1

#Splitting Data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)





We must also conduct a bit of EDA to understand the spread of data as well as to understand how much data we have available for each column, as well to see how the age is distributed
Based on what was seen I concluded that there was not much need for feature engineering

train = x_train.join(y_train)
train.info()
#train
train.hist(figsize=(15, 8))





This heatmap was very interesting to look at as although almost all variables did not have much correlation to er_visit it seems that date had a MASSIVE impact compared to everything else.

import seaborn as sns
plt.figure(figsize=(15, 8))
sns.heatmap(train.corr(), annot=True)





After looking through the equation that I was given as well as the nature of er_visit that logistic regression that is regularized(L2) by ridge regression is the most effective model.
There also exists the ridge regression model itself within scikit-learn which I have included in the bottom however this is mostly used for continuous target variables.
Er_visit is classification and logistic regression regularized by ridge regression (L2) is the most appropriate model. 

Ridge regression in the context of L2 is a means of fitting the data, there is both lasso and ridge. In the context of this model we utilize L2 which is a method of affecting the coefficients within our model. 
Based on our previous observation of the date coefficient having such a large correlation it would be useful to use L2 to penalize this coefficient within the regression model among others, 
ridge regression equally affects all variables, you can also adjust the magnitude by changeing the alpha within the model fit.

Ridge regression helps with multicollinearity and parameter tuning by equally distributing a penalization to predictor variables, stabilizing the model and making less noise within the model making it more efficient and less biased.

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

#target variable is usually not scaled
x_train, y_train = train.drop(['er_visit'], axis= 1), train['er_visit']
x_train_s = scaler.fit_transform(x_train)

#Below we have applied L2 Ridge regularization method to penalize coefficents
model = LogisticRegression(penalty='l2', solver='liblinear')
model.fit(x_train, y_train)








I am a little bit confused when the assignment has asked me to provide a 'ridge regression' as this is a classification problem(understanding er_visit binary classification). 
For this iteration I will provide logistic regression with L2 ridge regularization to fit the data. 
I will also provide a commented out Ridge model below, if this is the appropriate model you are seeking. 
In my opinion regularized(Ridge L2) logistic regression is the most appropriate.

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score

pred = model.predict(x_test)

accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
roc = roc_auc_score(y_test, pred)

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("Area Under ROC: ", roc)
Accuracy:  0.9885823025689819
Precision:  0.9754601226993865
Recall:  0.9520958083832335
Area Under ROC:  0.9737854607527028

#coeffs
coefficients = model.coef_
for i, coef in enumerate(coefficients[0]):
    print(f"Coefficient for feature {i}: {coef}")

Coefficient for feature 0: -0.006426792090723382
Coefficient for feature 1: 0.0015139757906968909
Coefficient for feature 2: 0.09029655764465556
Coefficient for feature 3: -0.0002476888256883041
Coefficient for feature 4: -0.0003076483980081201
Coefficient for feature 5: -0.000369095181584745
Coefficient for feature 6: -0.0006782584322027229
Coefficient for feature 7: -0.00023833465967282785
Coefficient for feature 8: -0.0003889008501202791


This model will not work as it is based off understanding continuous variables, 
our target variable is based on classification and thus will not work with the below model. 
Only with the application of L2 to logisitic regression.


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)
ridge.fit(x_train, y_train)
pred2 = ridge.predict(x_test)