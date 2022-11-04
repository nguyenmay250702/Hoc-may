from sympy import *
from math import *

###B1: 
"""
def grad(x):
    return x-1

def GD(x, eta):
    while abs(grad(x)) > eta:
        x = x - grad(x)
    return x

print("x = ", GD(126, 0.00001))
"""


###B2: Cho hàm số f(x)= x^2 + 5*sin(x). giải pt f'(x)=0

#tính F tại điểm x
def F(x):
    return x**2 +5*sin(x)

#tính đạo hàm tại 1 điểm 
def grad(x):
    return 2*x + 5*cos(x)

#tìm giá trị của x để f'(x)~0 
#k= số lần lặp, x= giá trị bắt đầu, eta= giá trị ~0 nhất có thể
def GD(x, eta,k):   
    i=0
    while (abs(grad(x)) > eta) and (i<=k):
        x = x - grad(x)
        i = i+1
    return x

x = GD(2, 0.001,5)

print("Giá trị làm cho f'(x)~0 là x = ", x)
print("f'({}) = ".format(x), grad(x))



"""
#tính đạo hàm của 1 hàm số
def derivative():
    x = Symbol('x')
    y = x**2 + 5*sin(x)
    y_x = diff(y, x)
    return y_x
"""










