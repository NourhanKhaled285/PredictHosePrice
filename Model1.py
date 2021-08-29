import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics


data=pd.read_csv("house_data.csv")
print(data.describe())
x=data["bedrooms"]
y=data["price"]

lin_mod = linear_model.LinearRegression()

x=np.expand_dims(x, axis=1)
y=np.expand_dims(y, axis=1)

lin_mod.fit(x,y)
ypred= lin_mod.predict(x)


mean_sq_error_bed=metrics.mean_squared_error(y,ypred)
print("the mean squared error when x=bedrooms is: ",mean_sq_error_bed)




x=data["sqft_lot"]
y=data["price"]

lin_model=linear_model.LinearRegression()
x=np.expand_dims(x,axis=1)
y=np.expand_dims(y,axis=1)
lin_model.fit(x,y)
ypred=lin_model.predict(x)
mean_sq_error_lot=metrics.mean_squared_error(y,ypred)
print("the mean squared error when x= is:sqft_lot ",mean_sq_error_lot)


