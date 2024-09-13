import pandas as pd
from fun import Game
from xai import shapley_value, banzhaf_value, core_lp, kernel,nash_barg_solution
from prettytable import PrettyTable


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
    #sanity check, imiting general ML model
    if len(x)!=2:
        raise Exception("Error the number of variables does not match the ones in the model definition")
      
    g=x[0]
    d=x[1]
    return g+d-2*g*d

def admitted(x):
    """Rule says that applicant can proceed with enrollment if f(g,d)>0"""
    return f(x)>0


if __name__=="__main__":
    DATA_PATH = "data.csv"
    data = pd.read_csv(DATA_PATH)
    #from admission test example, we drop f (groundtruth column)
    x_test = data.drop(['f'], axis=1)
    row_i = 2
    #print("prediction for g=1, d=1 is ",f(x_test.iloc[row_i]))
    #we create a game for each specific observation (row)
    first_game = Game(x_test.iloc[row_i], f)
    print('N=',first_game.grandcoalition.N)
    print('|N|=',first_game.grandcoalition.size)
    print('xN=',first_game.grandcoalition.xN)
    print('n PLAYERS Payoff',f(x_test.iloc[row_i]))
    S = set([1])
    print('Coalition value',first_game.v(S))
    #sv = shapley_value(first_game)
    #print('exact_shapley:',sv)
    
    #visual settings
    table = PrettyTable()
    red_rows = []
    table.field_names = ["players-values", "f(g,d)" , "admitted" ,"convex_game", "superadditive_game","exact_shapley", "exact_banzhaf", "lp_core", "core_set","nucleolus", "kernel", "nash_barg_sol", "weight_nash_barg_sol"]
    for i in range(len(x_test)):
        row = x_test.iloc[i]
        game = Game(row, f)
        convex = False
        core_res, core_set = core_lp(game)
        nucleolus = 0
        nash_barg_sol = 0
        w_nash_bargs_sol = 0
        alpha = [0.3, 0.7]
        egalitarian = 0
        table.add_row([row.to_dict(), f(row), admitted(row), game.is_convex(), game.is_superadditive(), shapley_value(game), banzhaf_value(game), core_res, core_set,nucleolus, kernel(game) ,nash_barg_solution(game), nash_barg_solution(game, alpha)])
        if not(game.is_convex()) or not(game.is_superadditive()):
            red_rows.append(len(table.rows) - 1)
    
    print(table)
    df = pd.DataFrame(table.rows, columns=table.field_names)
    df.to_csv("out/table.csv", index=False)

