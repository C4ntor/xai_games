import pandas as pd
import sklearn.datasets
import sklearn.linear_model
import sklearn.svm
from fun import Game
from xai import shapley_value, banzhaf_value, core, kernel,nash_barg_solution, nucleolus
from prettytable import PrettyTable
import sklearn
from sklearn.model_selection import train_test_split
import shap
import numpy as np


def compare(data_path, seed):
    df = pd.read_csv(data_path)
    X = df[['X', 'Z']]  # Independent variables
    y = df['Y']  # Dependent variable
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.5, random_state=seed)
    g=sklearn.linear_model.LinearRegression()  #svm.SVR()
    g.fit(X_train, y_train)


    def f(x):
        """
        Class representing any ML model, that takes a well-structured input (vector/matrix/.... of fixed size) and returns a well-structured output.

        @reference: Shapley Residuals: Quantifying the limits of the
        Shapley value for explanations (by Kumar, Scheidegger, Venkatasubramanian, Friedler)

        In this specific example, we assume x is a vector that contains just two elements: (g,d)


        g: (int) gender of the applicant, can be (-1) or (1)
        d: (int) department chosen by the applicant, can be (-1) or (1)

        it returns 
        x:  (int) where x is obtained as g+d-2*d*g  
        """
        
        return g.predict(x)




    x_test = X_test
    #print("prediction for g=1, d=1 is ",f(x_test.iloc[row_i]))
    #we create a game for each specific observation (row)
    #desc_game = Game(X_test.iloc[row_i:row_i+1], f,0, X_train)
    #print('N=',desc_game.grandcoalition.N)
    #print('|N|=',desc_game.grandcoalition.size)
    #print('xN=',desc_game.grandcoalition.xN)
    #sv = shapley_value(first_game)
    #print('exact_shapley:',sv)
    #visual settings
    table = PrettyTable()
    f_train_mean = y_train.mean()  #prev. was = 0
    table.field_names = ["players-values", "f(x)","convex","super_additive","exact_shapley", "kernel_shap", "exact_banzhaf", "core","nucleolus", "kernel", "nash_barg_sol", "weight_nash_barg_sol"]  #players-values is row in whole dataset: print(X_test.iloc[row_i])
    for i in range(0,len(y_test)):
        row = X_test.iloc[i:i+1]
        exSHAP = shap.KernelExplainer(g.predict, shap.sample(X_train,100))
        game = Game(row, f, f_train_mean, X_train)


        table.add_row([row.to_dict(),f(row),game.is_convex(),game.is_superadditive(),shapley_value(game), exSHAP.shap_values(row),banzhaf_value(game), core(game),nucleolus(game), kernel(game), nash_barg_solution(game, [0]*game.grandcoalition.size), nash_barg_solution(game, [f_train_mean]*game.grandcoalition.size)])

    df = pd.DataFrame(table.rows, columns=table.field_names)
    df.to_csv("out/table.csv", index=False)

