#assume a simple causal model : X->Y, Z->X, Z->Y
#e.g. Education(X) affects directly Income (Y), Work experience (Z) affect indirectly education and Income

#specify functional form for eachg relationship
#for now, let's go linear

#Y = b0+b1X+b2Z + error term

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


np.random.seed(42)
n = 1000  # number of data points
Z = np.random.normal(0, 1, n)  
epsilon_X = np.random.normal(0, 0.5, n)  # noise for X
X = np.random.normal(0, 1, n) + epsilon_X  
epsilon_Y = np.random.normal(0, 2, n)  # noise for Y
Y = 3 + 2 * X + 1.5 * Z + epsilon_Y  # Y = 3 + 2*X + 1.5*Z + noise

data = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})
plt.scatter(data['X'], data['Y'])
plt.xlabel('Education (X)')
plt.ylabel('Income (Y)')
plt.title('Generated Data: Education vs. Income')
plt.show()

plt.scatter(data['Z'], data['Y'])
plt.xlabel('Work Exp (Z)')
plt.ylabel('Income (Y)')
plt.title('Generated Data: Work Exp vs. Income')
plt.show()


X = data[['X','Z']]  
Y = data['Y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, Y_train)

# Step 5: Make predictions on the test data
Y_pred = model.predict(X_test)

# Step 6: Evaluate the model's performance
mse = mean_squared_error(Y_test, Y_pred)
#print(f"Mean Squared Error: {mse}")

# Coefficients and intercept of the model
print(f"Regression Coefficients (b1, b2): {model.coef_}")
print(f"Intercept (b0): {model.intercept_}")
data.to_csv('linear_scm.csv', index=False)


#non linear:
Z = np.random.normal(0, 1, n)

# Simulate error terms for X and Y
epsilon_X = np.random.normal(0, 0.5, n)
epsilon_Y = np.random.normal(0, 2, n)

# Apply ReLU function on Z to get X
X = np.maximum(0, 0.5 * Z) + epsilon_X  # X = ReLU(α1*Z) + noise
#it's clear that increasing Z, will increase X...


# Non-linear relationship for Y: 
Y = 3 + np.exp(X) -10 * sigmoid(Z) + epsilon_Y 
#clear that Y,X are + corr, and Y,Z are - corr
#but which one influences more Y? 
#X should have more influence on Y (both when increasing and when decreasing)

# Create a DataFrame
data = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})
plt.scatter(data['X'], data['Y'])
plt.xlabel('Education (X)')
plt.ylabel('Income (Y)')
plt.title('Generated Data: Education vs. Income')
plt.show()

plt.scatter(data['Z'], data['Y'])
plt.xlabel('Work Exp (Z)')
plt.ylabel('Income (Y)')
plt.title('Generated Data: Work Exp vs. Income')
plt.show()
X = data[['X','Z']]  
Y = data['Y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, Y_train)

# Step 5: Make predictions on the test data
Y_pred = model.predict(X_test)

# Step 6: Evaluate the model's performance
mse = mean_squared_error(Y_test, Y_pred)
#print(f"Mean Squared Error: {mse}")

# Coefficients and intercept of the model
print(f"Regression Coefficients (b1, b2): {model.coef_}")
print(f"Intercept (b0): {model.intercept_}")
data.to_csv('nonlinear_scm.csv', index=False)