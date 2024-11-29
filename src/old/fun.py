import itertools
import random
import numpy as np
import pandas as pd
random.seed(0)


class GrandCoalition:
    """
    Creates the all-players coalition for the game. Following the order of features in input row, it assigns at each feature an index, and inizializes an hashmap between player indices and associated feature value
    (indices are between 0 and n-1) where n is the number of players
    """
    def __init__(self, x, X_train):
        self.xN = {i: value for i, value in enumerate(x)}
        self.N = set(self.xN.keys())
        self.size = len(self.N)
        self.X_train = X_train

    def sample_from_tr_distr(self, i):
        return np.random.choice(self.X_train[self.xN[i]])

class Game:
    """
    A class resembling a cooperative game.

    Args:
        @x: (pandas.Series) represent a row containing observation values of input features (X's) of our dataset. (Y, the groundtruth is excluded)
                            e.g. age        20
                                 income     300

        @payoff_function: (function) the value (payoff) function. Can later be applied on a subset (S) (of grandcoalition N). Recall N will just be a set of indices

        @disagreement_point: (vector)  used in Bargaining Problems. Contains utilities/payoff values of each player, whenever an agreement between player is not reached. 
    """

    def __init__(self, x, payoff_function, f_train_mean, X_train, disagreement_point=None) -> None:
        self.grandcoalition = GrandCoalition(x, X_train)
        self.obs = x
        self.f_train_mean = f_train_mean
        #The input size accepted by v, must match all player coalition 
        self._payoff_function = payoff_function  #maybe another layer of abstraction to handle missing features?
        if disagreement_point ==None:
            self.disagreement_point= [0]* self.grandcoalition.size
        else:   
            if len(disagreement_point)!=self.grandcoalition.size:
                raise Exception("The disagreement point must be of same size as the grandcoalition")


    def build_input_n(self, coalition, mode):
        """
        Given a coalition, it will build the corresponding vector, using the values of features in the coalition, and following mode rule for excluded ones.

        Args:
            @coalition: (set)   set of indices of players that form the coalition
            @mode:  (int)       can be 0 or 1. If 0, will put excluded feature values to 0. If 1 will replace it with random values.

        """
        #for testing purpose we force mode=1
        mode=1

        #we assume that excluding features means put excluded feature values to 0  (mode=0)
        if mode==0:
            vector =  [0] * self.grandcoalition.size
        else:
            vector =  [0] * self.grandcoalition.size
            for i in self.grandcoalition.N:
                vector[i] = self.grandcoalition.sample_from_tr_distr(i) #sample randomly from training distr of var i

        for i in coalition:
                    vector[i] = self.grandcoalition.xN[i]  #replace present feature with their actual values from current obs
        return vector

    def v(self, coalition, mode=0):
        """
        This function enables to apply the payoff function (f) (in ML settings this will be the model itself) to be applied on any subcoalition of N.
        It does this, without the need of modify the ML model structure. The model can still accept the same structured input.
        payoff_function originally designed for vector of size n, will be applied on the new constructed vector, that following the mode rule, excludes absent features.

        Args:
            @coalition: (set)
            @mode:  (int)

        """
        #ASSUMPTION: what does it mean to have v({}) in ML? We assume that v(empty_set)=E(f(x))
        if len(coalition)==0:
            return self.f_train_mean
        input_size_n = self.build_input_n(coalition, mode) 
        #replace labels values with actual values in the observation   
        Y = [0]*len(input_size_n)


        for i, label in enumerate(input_size_n):
            if isinstance(self.obs, pd.DataFrame):  # If it's a DataFrame
                if label in list(self.obs.columns):
                    Y[i]= self.obs[label].values[0]
                    # Do something for DataFrame case
            elif isinstance(self.obs, pd.Series):  # If it's a Series
                if label in self.obs.index:
                    print(self.obs)
                    exit()
        
        Y = pd.Series(Y)

        labels = [self.grandcoalition.xN[i] for i in range(len(Y))]
        Y.index = labels
        Y = Y.values.reshape(1, -1)
        return self._payoff_function(Y)

   

    def is_convex(self):
        """
        It checks if game is convex, that is:  for any A,B subset of N.  v(A.union.B) >= v(A)+v(B)-v(A.intersection.B) 
        """
        subsets = list(itertools.chain.from_iterable(itertools.combinations(self.grandcoalition.N, r) for r in range(self.grandcoalition.size + 1)))
        subsets = [set(subset) for subset in subsets]
        for A in subsets:
            for B in subsets:
                union_ab = A.union(B)
                intersection_ab = A.intersection(B)
                
                if self.v(union_ab) < self.v(A) + self.v(B) - self.v(intersection_ab):
                    return False
        return True

    def is_superadditive(self):
        """
        It checks if game is convex, that is:  for any two disjoint subsets A,B of N.  v(A.union.B) >= v(A)+v(B) 
        """
        subsets = list(itertools.chain.from_iterable(itertools.combinations(self.grandcoalition.N, r) for r in range(self.grandcoalition.size + 1)))
        subsets = [set(subset) for subset in subsets]
        for A in subsets:
            for B in subsets:
                if (A.isdisjoint(B)):
                    union_ab = A.union(B)
                    
                    # Check if superadditivity holds
                    if self.v(union_ab) < self.v(A) + self.v(B):
                        return False
        return True



