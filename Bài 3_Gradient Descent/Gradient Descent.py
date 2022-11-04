# Tìm điểm cực trị cục bộ của hàm f(x) = x**2 + 5*sin(x)

import numpy as np
#grad(x): Tính đạo hàm của hàm f(x)
def grad(x):
    return 2*x + 5*np.cos(x)

#cost(x): Tính giá trị của hàm f(x)
def cost(x):
    return x**2 + 5*np.sin(x)

#myGD1(eta, x0): Tìm điểm cực trị cục bộ của hàm f(x) theo công thức Gradient Descent: x(t+1) = x(t) - eta*grad(x)
#eta là learning rate (tốc độ học), x0 là điểm khởi tạo

def myGD1(eta, x0):
    x = [x0]
    for it in range(100):
        x_new = x[-1] - eta*grad(x[-1])
        if abs(grad(x_new)) < 1e-3:
            break
        x.append(x_new)
    return (x, it)
(x1, it1) = myGD1(.1, -5)
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1[-1], cost(x1[-1]), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2[-1], cost(x2[-1]), it2))