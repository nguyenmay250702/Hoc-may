from numpy import *
from sklearn import datasets, linear_model

X = array([[1,-2,0],[2,5,1]])
Y = array([1,6,1])

"""
#cách cũ:
W_T = (linalg.pinv(X.T@X))@X.T@Y

W = W_T.T
print("W = ", W)
print("Giá = ",(X.T)@W)
#print("f'(w) = ", X.T@X@W_T - X.T@Y)
"""

W = linalg.pinv(X@X.T)@X@Y
print("W = ", W)
print("dự đoán: ", X.T@W)

#sd sklearn
"""
# fit the model by Linear Regression
regr = linear_model.LinearRegression()
regr.fit(X, y) # in scikit-learn, each sample is one row
# Compare two results
print("scikit-learn’s solution: w_1 = ", regr.coef_[0], "w_0 = ",\
regr.intercept_)
print("our solution : w_1 = ", w[1], "w_0 = ", w[0])


"""
