from __future__ import print_function
import numpy as np
#import matplotlib.pyplot as plt


# height (cm), input data, each row is a data point
X = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T

# weight (kg)
y = np.array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

# Building Xbar
#X.shape[0]: lấy ra số hàng của ma trận X(= 13)
one = np.ones((X.shape[0], 1))  #tạo ra ma trận toàn số 1 có số hàng là 13 và số cột là 1
Xbar = np.concatenate((one, X), axis = 1) #nối ma trận one và ma trận x theo thứ tự [one][X], axis=1: nối theo chiều ngang

# Calculating weights of the linear regression model
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)
# weights
w_0, w_1 = w[0], w[1]

y1 = w_1*155 + w_0
y2 = w_1*160 + w_0
print("Input 155cm, true output 52kg, predicted output %.2fkg." %(y1) )
print("Input 160cm, true output 56kg, predicted output %.2fkg." %(y2) )

print("\Xbar = ", Xbar)
print("\nXbar[m,n] = ", Xbar.shape)
print("\nW = ", w)



"""
#test
from numpy import *

# height (cm), input data, each row is a data point
X1 = array([147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183])
X2 = array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T
X3 = array([[147], [150], [153], [158], [163], [165], [168], [170], [173], [175], [178], [180], [183]])

# weight (kg)
y = array([49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68])

print("X2-X3 = \n", X2 - X3)

print("\ncỡ X1: ",X1.shape)
print("- X1*y = ", X1@y)


print("\ncỡ X2",X2.shape)
print("- X2*y = ", X2@y)

print("\ncỡ X2",X3.shape)
print("- X3*y = ", X3@y)
"""





















































