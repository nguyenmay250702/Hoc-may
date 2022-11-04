from math import *

def grad(x):
    return 4*(x**3) + 10*sin(x)*cos(x)

def GD(x, eta, k):
    i=0
    while abs(grad(x)) > eta and i<k:
        x = x - grad(x)
        i = i+1
    return x

x = GD(5, 0.0001, 4)
print("Giá trị để f'(x)~0 là x = ", x)
print("f'({}) = ".format(x), grad(x))

