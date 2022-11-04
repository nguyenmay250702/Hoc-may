from numpy import *

X = array([[60,2,10],[40,2,5],[100,3,7]])
Y = array([10,12,20])

W = linalg.pinv(X@(X.T))@X@Y

print("\nW = ",W)

print("Giá dự đoán1 = ", X.T@W)

R = array([[60,2,10]])
print("Giá dự đoán2 = ", R@W)


"""
X1 = array([60,2,10])
print("thong tin: ",X1)
print("y = ", X1.T@W)
"""

"""
X = 60  2  10
    40  2  5
    100 3  7

Y = 10
    12
    20
    
W = -2.45238095
    15.23809524
    -4.52380952

"""
