import pandas as pd

# Read your csv file
df = pd.read_csv('Salary_Data.csv')

#Divide Dataset into two parts, i.e, dependent(y) and independent variables(x)
x=df.iloc[:,:-1].values 1st colon is all the rows and all columns except the last one
y=df.iloc[:,1].values

#Split the data into training set and testing set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)


#implement linear regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

y_predict = lr.predict(x_test)

# Graphs
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train)
plt.plot(x_train,lr.predict(x_train))

