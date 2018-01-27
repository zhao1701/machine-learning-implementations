import numpy as np

# define Ridge Regression class

class RidgeRegression:
   
   lambd = None # regularization parameter
   weights = None # the weights to be solved that minimize the loss
   n_features = None 
   df = None # the degrees of freedom of the model, a function of lambd
   method = None # normal or svd
   
   def __init__(self, lambd = 3, method = 'svd'):
       self.lambd = lambd
       self.method = method
   
   def fit(self, X, y):
       self.n_features = X.shape[1]
       if self.method == 'normal':
           # solve for weights w
           first = np.linalg.inv(self.lambd * np.eye(self.n_features) + np.dot(X.T,X))
           second = np.dot(X.T, y)
           self.weights = np.dot(first, second)
           
           # calculate degrees of freedom using previously computed 'first' variable
           df_matrix = X.dot(first).dot(X.T)
           self.df = np.trace(df_matrix)
       elif self.method == 'svd':
           # the svd function returns V^T as V, which should be corrected for clarity
           # S is a vector of singular values rather than a diagonal matrix
           U, S, V = np.linalg.svd(X, full_matrices=False)
           V = V.T 
           S_lambda_inv = S / (self.lambd + S**2) # also a vector of values
           S_lambda_inv_diag = np.diag(S_lambda_inv) # convert vector to diagonal matrix
           self.weights = V.dot(S_lambda_inv_diag).dot(U.T).dot(y)
           self.df = np.sum(S**2 / (self.lambd + S**2))
       return self      
       
   def predict(self, X):
       return np.dot(X, self.weights)

def rmse(y, y_pred):
   mse = np.mean((y - y_pred)**2)
   rmse = np.sqrt(mse)
   return rmse