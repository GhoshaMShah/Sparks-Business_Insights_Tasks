import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data.plot(x='Hours', y='Scores', style='o')
plt.title('Hours v/s Score')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.25, random_state = 0)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test.reshape(-1,1))
NumberOfHours = 9.25
ans = model.predict([[NumberOfHours]])
print('Score for',NumberOfHours,'=',ans)

from sklearn import metrics
print('Mean Absolute Error =',metrics.mean_absolute_error(Y_test, Y_pred))