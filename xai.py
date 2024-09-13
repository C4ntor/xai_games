import itertools
import math
import numpy as np
import pulp
import random
from scipy.optimize import minimize
random.seed(0)

def shapley_value(game):
    """
    Computes exact shapley value (contributions) for each player in the game
    Args:

    @game: (Game) an instance of Game Class 
    """
    n = game.grandcoalition.size
    all_players = game.grandcoalition.N
    shapley_values = [0] * n
    
    for i in all_players:
        all_other_players = game.grandcoalition.N.copy()
        all_other_players.remove(i)
        for coalition in itertools.chain.from_iterable(itertools.combinations(all_other_players, r) for r in range(n)):
           coalition_set = set(coalition)
           v_with_i = game.v(coalition_set.union({i}))
           v_without_i = game.v(coalition_set)
           marginal_contribution = v_with_i - v_without_i
           #print("coalition",coalition_set)
           #print("coalition_with_i", coalition_set.union({i}))
           #print("v_with_i", v_with_i)
           #print("v_without_i", v_without_i)
           #print("marg", marginal_contribution)
           #print(f'marginal_contr_for_{i}_is',marginal_contribution)
           weight = math.factorial(len(coalition)) * (math.factorial(n - len(coalition) - 1))
           #print("weight", weight)
           shapley_values[i] += weight*marginal_contribution
        shapley_values[i] = shapley_values[i]/ math.factorial(n)
    return shapley_values

def banzhaf_value(game):
    """
    Computes exact banzhaf value (contributions) for each player in the game
    
    Args:
        @game (Game) an instance of Game Class 
    """
    n = game.grandcoalition.size
    all_players = game.grandcoalition.N
    banzhaf_value = [0] * n
    
    for i in all_players:
        all_other_players = game.grandcoalition.N.copy()
        all_other_players.remove(i)
        for coalition in itertools.chain.from_iterable(itertools.combinations(all_other_players, r) for r in range(n)):
           coalition_set = set(coalition)
           v_with_i = game.v(coalition_set.union({i}))
           v_without_i = game.v(coalition_set)
           marginal_contribution = v_with_i - v_without_i
           #print("coalition",coalition_set)
           #print("coalition_with_i", coalition_set.union({i}))
           #print("v_with_i", v_with_i)
           #print("v_without_i", v_without_i)
           #print("marg", marginal_contribution)
           #print(f'marginal_contr_for_{i}_is',marginal_contribution)
           #print("weight", weight)
           banzhaf_value[i] += marginal_contribution
        banzhaf_value[i] = banzhaf_value[i]/ (2**(n-1))
    return banzhaf_value


def core_lp(game, solutions_limit=10, epsilon = 1e-4):
    """
    Defines and solves LP to find the Core
    We define a dummy objective function (Min 0) sbj to constraints of individual rationality, coalitional rationality, and efficiency (feasibility could have also been used)

    In case of multiple optimal solutions, Only one solution is shown by the solver.
    In that case, we add a small (random) perturbation to the objective function in order to get another solution, without modifying the feasible set.
    """

    all_players = game.grandcoalition.N
    n = game.grandcoalition.size

    problem = pulp.LpProblem("Core", pulp.LpMinimize)

    lp_vars = [pulp.LpVariable(f"x{i}") for i in all_players]
    
    problem += 0

    problem+= pulp.lpSum(lp_vars) == game.v(all_players)
    #ASSUMPTION: We put no constraint on the sign of contributions, it means they can also be negative.
    # for var in lp_vars:
    #     problem += var >=0

    for coalition_size in range(0, n + 1):
        for coalition in itertools.combinations(all_players, coalition_size):
            coalition_set = set(coalition)
            coalition_value = game.v(coalition_set)
            var_in_coalition = [var for var in lp_vars if var.name in [f"x{i}" for i in coalition_set]]
            problem += pulp.lpSum(var_in_coalition) >= coalition_value

    #print(problem)
    solutions = []
    status = problem.solve()
    status = pulp.LpStatus[problem.status]

    if status == 'Optimal':
        first_solution = [var.varValue for var in lp_vars]
        solutions.append(first_solution)

        num_constraints = len(problem.constraints)
        basic_vars = sum([1 for var in lp_vars if var.varValue != 0])
        
        if basic_vars < num_constraints:
            status="Multiple solutions"

            for _ in range(solutions_limit):
                perturb_problem = problem.copy()
                perturb_obj = pulp.lpSum([random.uniform(-epsilon, epsilon) * lp_var for lp_var in lp_vars])
                perturb_problem += perturb_obj
                #print("pert_pr",perturb_problem)
                new_status = perturb_problem.solve()
                new_status_str = pulp.LpStatus[perturb_problem.status]
                if new_status_str == 'Optimal':
                    perturbed_solution = [var.varValue for var in lp_vars]
                    
                    # Check if the new solution is different from previous ones
                    if perturbed_solution not in solutions:
                        solutions.append(perturbed_solution)
          
    return status, solutions



def nash_barg_rule(x,d, alpha):
    product= 1
    for i in range(len(d)):
        product = product * (x[i] - d[i]) ** alpha[i]
    return -product

def constraint_barg(x,d):
    return x-d

def constraint_ind_rat(x,v):
    return x-v

def constraint_eff(x, grandpayoff):
    return np.sum(x) - grandpayoff
    
def nash_barg_solution(game, alpha=None):
    """
    Define an optimization problem, to find x s.t. weighted nash bargaining rule is maximized, sbj to constraints of efficiency (or feasibility), individual rationality and bargaining constraint.
    @game: (Game)
    @alpha: (float)  is the exponent of nash bargaining rule (expressing the power of each player). If None, the rule computes (symmetric) nash bargaining solution.
    """

    d = np.array(game.disagreement_point)
    v = [game.v(set({i})) for i in game.grandcoalition.N]
    grandpayoff = game.v(game.grandcoalition.N)

    if alpha==None:
        alpha = [1/len(d)]*len(d)

    initial_value = max(np.max(d), np.max(v)) + 1  #ensure x0 greater than d, v
    x0 = initial_value * np.ones(len(d))

    cons = [
    {'type': 'ineq', 'fun': lambda x: constraint_ind_rat(x, v)},  # x_i >= v(i)
    {'type': 'eq', 'fun': lambda x: constraint_eff(x, grandpayoff)},   # Sum of x_i = v(N)
    {'type': 'ineq', 'fun': lambda x: constraint_barg(x, d)} # x>d
    ]


    result = minimize(nash_barg_rule, x0, args=(d, alpha), constraints=cons)


    if result.success:
        optimal_x = result.x
        #optimal_value = -result.fun
        return optimal_x
    else:
        return "ERR:"+result.message


def kernel(game):
    prob = pulp.LpProblem("Kernel", pulp.LpMinimize)
    n = game.grandcoalition.size
    x = pulp.LpVariable.dicts("x", (range(n), range(n)), lowBound=0, cat='Continuous')
    return "ERR"
        
