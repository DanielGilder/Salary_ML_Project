import joblib
import numpy as np
loaded_model = joblib.load("model.joblib")

while True:
    years = float(input("Please enter the predicted salary after X years  :"))
    print(loaded_model.predict(np.array([[years]])))
    print("\n")