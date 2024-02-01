import pandas as pd 

#importing the dataset
data = pd.read_csv("data/Real_Estate.csv")

#importing all the necessary libraries associated to model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

#setting columns for features and target
features = ['Distance to the nearest MRT station', 'Number of convenience stores', 'Latitude', 'Longitude']
target = 'House price of unit area'

#setting x and y
x = data[features]
y = data[target]

#splitting the data into training and testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=42)

#calling the model
model = LinearRegression()

#training
model.fit(x_train,y_train)