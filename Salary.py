import csv
import numpy as np

from sklearn import linear_model

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

X_training = xdata[:30]
Y_training = ydata[:30]

X_test = xdata[31:].reshape(-1,1)
Y_test = ydata[31:]

model = linear_model.LinearRegression()

model.fit(X_test, Y_test)

input_val = np.array([[10]])

print("The predicted salary is:")
print(model.predict(input_val))