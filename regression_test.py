import pandas as pd
import sklearn.datasets
import sklearn.linear_model
import sklearn.svm
from fun import Game
from xai import shapley_value, banzhaf_value, core_lp, kernel,nash_barg_solution, nucleolus
from prettytable import PrettyTable
import sklearn
from sklearn.model_selection import train_test_split
import shap
import numpy as np

df = pd.read_csv('linear_scm.csv')
X = df[['X', 'Z']]  # Independent variables
y = df['Y']  # Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
row_i = 1 #first test row
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
    global g
    return g.predict(x)



if __name__=="__main__":
    #from admission test example, we drop f (groundtruth column)
    x_test = X_test
    #print("prediction for g=1, d=1 is ",f(x_test.iloc[row_i]))
    #we create a game for each specific observation (row)
    desc_game = Game(X_test.iloc[row_i:row_i+1], f,0, X_train)
    print('N=',desc_game.grandcoalition.N)
    print('|N|=',desc_game.grandcoalition.size)
    print('xN=',desc_game.grandcoalition.xN)
    #sv = shapley_value(first_game)
    #print('exact_shapley:',sv)
    #visual settings
    table = PrettyTable()
    red_rows = []
    f_train_mean = y_train.mean()  #prev. was = 0
    table.field_names = ["players-values", "f(x)","super_additive","exact_shapley", "kernel_shap", "exact_banzhaf", "lp_core", "core_set","nucleolus", "kernel", "nash_barg_sol", "weight_nash_barg_sol"]  #players-values is row in whole dataset: print(X_test.iloc[row_i])
    for i in range(row_i,row_i+1):
        exSHAP = shap.KernelExplainer(g.predict, shap.sample(X_train,100))
        kernel_shap = exSHAP.shap_values(X_test.iloc[row_i:row_i+1])
        #shap.summary_plot(kernel_shap, X_test.iloc[row_i:row_i+1])
        row = X_test.iloc[row_i:row_i+1]
        game = Game(row, f, f_train_mean, X_train)
        convex = False
        core_res, core_set = core_lp(game)
        nash_barg_sol = 0
        w_nash_bargs_sol = 0
        egalitarian = 0
        table.add_row([row.to_dict(),f(X_test.iloc[row_i:row_i+1]),game.is_superadditive(),shapley_value(game), kernel_shap,banzhaf_value(game), core_res, core_set,nucleolus(game), kernel(game), nash_barg_solution(game, [0]*desc_game.grandcoalition.size), nash_barg_solution(game, [f_train_mean]*desc_game.grandcoalition.size)])
        if not(game.is_convex()) or not(game.is_superadditive()):
            red_rows.append(len(table.rows) - 1)
    
    print(table)
    df = pd.DataFrame(table.rows, columns=table.field_names)
    df.to_csv("out/table.csv", index=False)

   