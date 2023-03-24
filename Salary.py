import csv
import numpy as np
import joblib


from sklearn import linear_model
from sklearn.metrics import accuracy_score


with open('Salary.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    xdata = [float(row[0]) for row in reader]
    
with open('Salary.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader)
    ydata = [float(row[1]) for row in reader]


xdata = np.array(xdata)
ydata = np.array(ydata)

X_training = xdata[:30].reshape(-1,1)
Y_training = ydata[:30]

X_test = xdata[31:].reshape(-1,1)
Y_test = ydata[31:]

model = linear_model.LinearRegression()

model.fit(X_training, Y_training)

y_pred =  model.predict(X_test)


print("The predicted salary is:")
print(y_pred)

joblib.dump(model, 'model.joblib')

#print(accuracy_score(Y_test, y_pred))
