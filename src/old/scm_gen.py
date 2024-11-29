#assume a simple causal model : X->Y, Z->X, Z->Y
#e.g. Education(X) affects directly Income (Y), Work experience (Z) affect indirectly education and Income

#specify functional form for eachg relationship
#for now, let's go linear
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

seed=0
np.random.seed(seed)
n = 10  # number of data points

beta0 = np.random.normal(0,1,1)
beta1= np.random.normal(0,1,1)
beta2 = np.random.normal(0,1,1)


#relationship gen
epsilon_Z = np.random.normal(0, 0.5, n)
Z = np.random.normal(0, 1, n)  + epsilon_Z
epsilon_X = np.random.normal(0, 0.5, n)  # noise for X
X = np.random.normal(0, 1, n) + epsilon_X  
epsilon_Y = np.random.normal(0, 0.5, n)  # noise for Y
Y = beta0 + beta1 * X + beta2* Z + epsilon_Y  #Y = b0+b1X+b2Z + error term


data = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})
plt.scatter(data['X'], data['Y'])
plt.xlabel('(X)')
plt.ylabel('(Y)')
plt.title('Y,X')
plt.show()

plt.scatter(data['Z'], data['Y'])
plt.xlabel('(Z)')
plt.ylabel('(Y)')
plt.title('Y,Z')
plt.show()


X = data[['X','Z']]  
Y = data['Y']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=seed)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
data.to_csv('linear_scm.csv', index=False)


data_gen_specification = [
    ["beta0", "beta1", "beta2", "est_beta0", "est_beta1", "est_beta2", "rnd_seed"], 
    [beta0, beta1, beta2, model.intercept_, model.coef_[0], model.coef_[1], seed]]

with open("out/summary.csv", mode="w", newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data_gen_specification)

from explain import compare
compare("linear_scm.csv",seed)
