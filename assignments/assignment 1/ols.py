import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import norm, inv

def ols_min(y, X):
    
    def objfun_ols(b, y, X):
        r = y - np.dot(X,b)
        return norm(r)**2
    
    # Ideally, use some good approximation as a starting point
    b_init = np.asarray([0, 0, 0])
    # You can pass extra arguments to the objective function code using args
    est = minimize(objfun_ols, b_init, args = (y,X))
    return est.x


# Input and prepare data
df = pd.read_csv('hw1data.csv')

y = df.loc[:, 'logq'].to_numpy()
X = df.loc[:, ['logl', 'logk']]\
      .to_numpy()
X = np.append(np.ones([df.shape[0], 1]),X, axis=1)

# Find OLS estimates, use matrix formula to cross-check
b_ols = ols_min(y, X)
b_check = np.dot(inv(X.T@X), np.dot(X.T, y))

print('OLS coefficients, [b_0, b_l, b_k]')
print(f'OLS using the exact formula: {b_check}')
print(f'OLS using optimization: {b_ols}')