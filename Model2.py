import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model

from Model1 import *
import math





data=pd.read_csv("house_data.csv")
print(data.describe())

x=data["sqft_living"]
y=data["price"]

lin_model=linear_model.LinearRegression()
x=np.expand_dims(x,axis=1)
y=np.expand_dims(y,axis=1)
lin_model.fit(x,y)
ypred=lin_model.predict(x)




mean_sq_error_living=metrics.mean_squared_error(y,ypred)
print("the mean squared error when x=sqft_living is: ",mean_sq_error_living)




plt.scatter(x,y)

plt.xlabel("sqft_living",fontsize = 20)
plt.ylabel("price",fontsize = 20)

plt.plot(x,ypred,color='pink',linewidth=5)
plt.show()


x=data["bathrooms"]
y=data["price"]


lin_model=linear_model.LinearRegression()
x=np.expand_dims(x,axis=1)
y=np.expand_dims(y,axis=1)
lin_model.fit(x,y)
ypred=lin_model.predict(x)



mean_sq_error_bath=metrics.mean_squared_error(y,ypred)

print("the mean squared error when x=bathrooms is: ",mean_sq_error_bath,"\n")


plt.scatter(x,y)

plt.xlabel("bathrooms",fontsize = 20)
plt.ylabel("price",fontsize = 20)

plt.plot(x,ypred,color='pink',linewidth=5)
plt.show()


first_min=min(mean_sq_error_bath,mean_sq_error_living,mean_sq_error_lot,mean_sq_error_bed)
second_min=min(mean_sq_error_bath,mean_sq_error_lot,mean_sq_error_bed)

print("first minimum is : ",first_min)
print("second minimum is : ",second_min)







