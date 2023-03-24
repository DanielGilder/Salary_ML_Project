import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
data = pd.read_csv('Salary.csv')

# Create training data
X_train = np.array([[1], [2], [3]])
y_train = np.array([24, 44, 64])

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict with a single sample
X_new = np.array([[10]])
print(model.predict(X_new))

# Display prediction
#print(model.coef_)