import itertools
import math
import numpy as np
import pulp
import random
from scipy.optimize import minimize
from scipy.optimize import linprog
from itertools import combinations
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
    flatten_shapley_values= [float(item[0]) for item in shapley_values]
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


def core(game, solutions_limit=100, epsilon = 1e-4):
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




def constraint_barg(x,d):
    return x-d

def constraint_ind_rat(x,v):
    return x-v

def constraint_eff(x, grandpayoff):
    return np.sum(x) - grandpayoff
    
def nash_barg_solution(game, alpha):
    """
    Define an optimization problem, to find x s.t. weighted nash bargaining rule is maximized, sbj to constraints of efficiency (or feasibility), individual rationality and bargaining constraint.
    @game: (Game)
    @alpha: (float)  is the exponent of nash bargaining rule (expressing the power of each player). If None, the rule computes (symmetric) nash bargaining solution.
    """
    n = len(game.disagreement_point)
    d = game.disagreement_point
    v = [(float(game.v(set({i}))[0])) for i in game.grandcoalition.N]
    grandpayoff = game.v(game.grandcoalition.N)[0]


    def objective(x):
        return -np.prod([(x[i] - d[i])**alpha[i] for i in range(n)])

    # Define the constraint: the sum of the utilities must be equal to the total surplus
    def constraint(x):
        return np.sum(x) - grandpayoff

    initial_guess = np.full(n, grandpayoff / n)
    bounds = [(d[i], None) for i in range(n)]

    # Define the constraint as a dictionary for scipy's minimize function
    cons = {'type': 'eq', 'fun': constraint}
    
    # Use scipy's minimize function to find the Nash bargaining solution
    result = minimize(objective, initial_guess, bounds=bounds, constraints=cons)
    
    if result.success:
        return result.x
    else:
        return "Optimization failed"




def compute_excess(v, x, S):
    """Compute the excess for coalition S."""
    return v(S) - sum(x[i] for i in S)

def kernel(game, epsilon=1e-6, max_iter=1000):
    """
    Find the kernel of an n-player cooperative game.
    
    Parameters:
    - n: Number of players
    - v: Characteristic function (function accepting subsets as tuples)
    - epsilon: Convergence tolerance
    - max_iter: Maximum number of iterations
    
    Returns:
    - Payoff vector x in the kernel
    """
    # Initialize x equally
    n = game.grandcoalition.size
    v = game.v
    x = np.full(n, v(tuple(range(n))) / n)
    
    for iteration in range(max_iter):
        balanced = True
        
        # Iterate over all pairs of players
        for i in range(n):
            for j in range(i+1, n):
                # Compute maximum excesses for i->j and j->i
                e_ij = max(compute_excess(v, x, S) for S in powerset(range(n)) if i in S and j in S)
                e_ji = max(compute_excess(v, x, S) for S in powerset(range(n)) if i in S and j in S)
                
                # If excesses are unbalanced, adjust payoffs
                if abs(e_ij - e_ji) > epsilon:
                    balanced = False
                    delta = (e_ij - e_ji) / 2
                    x[i] -= delta
                    x[j] += delta
        
        if balanced:
            break
    
    return x

def powerset(iterable):
    """Generate all subsets of a given set."""
    s = list(iterable)
    return [tuple(comb) for r in range(len(s)+1) for comb in combinations(s, r)]


def nucleolus(game):
    """
    Compute the nucleolus of a cooperative game.

    Parameters:
    - v: Characteristic function, a dictionary where keys are coalitions (tuples) and values are worths.
    - n: Number of players.

    Returns:
    - Nucleolus payoff vector.
    """
    # Players
    n = game.grandcoalition.size
    v = game.v
    N = tuple(range(n))
    coalitions = list(powerset(N))[1:]  # All non-empty subsets
    
    # Initialize LP constraints
    A_eq = [np.ones(n)]  # Efficiency constraint: sum of payoffs equals v(N)
    b_eq = [v(N)]
    
    A_ub = []  # For coalition excess constraints
    b_ub = []
    
    # Individual rationality constraints
    for i in range(n):
        A_eq.append(np.zeros(n))
        A_eq[-1][i] = 1
        b_eq.append(v((i,)))
    
    # Add coalition constraints iteratively
    x = np.full(n, v(N) / n)  # Start with equal division
    
    while True:
        # Add excess constraints for all coalitions
        A_ub = []
        b_ub = []
        for S in coalitions:
            constraint = np.zeros(n)
            for i in S:
                constraint[i] = -1
            A_ub.append(constraint)
            b_ub.append(-v(S))
        
        # Add new variables to minimize the maximum excess
        c = np.zeros(n + 1)
        c[-1] = 1  # Objective: minimize the auxiliary variable
        
        A_ub = np.hstack([A_ub, -np.ones((len(A_ub), 1))])
        A_eq = np.hstack([A_eq, np.zeros((len(A_eq), 1))])
        
        # Solve the LP
        res = linprog(
            c, 
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=(None, None)
        )
        
        if res.success:
            x_new = res.x[:-1]
            z = res.x[-1]
            
            # Check termination condition
            if np.isclose(z, 0, atol=1e-6):
                break
            
            # Update constraints to refine solution
            excesses = [v(S) - sum(x_new[i] for i in S) for S in coalitions]
            max_excess = max(excesses)
            max_coalitions = [coalitions[i] for i, e in enumerate(excesses) if np.isclose(e, max_excess)]
            
            for S in max_coalitions:
                constraint = np.zeros(n)
                for i in S:
                    constraint[i] = -1
                A_eq = np.vstack([A_eq, np.hstack([constraint, 0])])
                b_eq.append(-v(S))
            
            x = x_new
        else:
            return ValueError("LP failed to converge")
    
    return x