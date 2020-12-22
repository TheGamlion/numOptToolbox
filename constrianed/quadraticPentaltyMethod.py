import numpy as np
from numpy.linalg import *
import matplotlib.pyplot as plt
from sympy.plotting import plot, plot3d, PlotGrid
from sympy import solve, Poly, Eq, Function, exp
from sympy import symbols, Matrix, Transpose
from sympy import *
from sympy.vector import CoordSys3D, gradient
from mpl_toolkits.mplot3d import Axes3D
from sympy import lambdify
import sympy.core.function
from sympy.matrices import Matrix, eye, zeros, ones, diag, GramSchmidt


def newton(function,x0,epsilon):
    grad = get_gradient(function,x0)
    hess = get_hessian(function,x0)
    d = -1*inv(hess).dot(grad)
    x_prev = x0
    x = x0 + d
    while np.abs(norm(x) - norm(x_prev)) >= epsilon:
        grad = get_gradient(function,x)
        hess = get_hessian(function,x)
        d = -1*inv(hess).dot(grad)
        x_prev = x
        x = x + d

    return x

def get_gradient(function, x):
    variables = list(ordered(function.free_symbols))
    Gradient = simplify(derive_by_array(function, variables))
    Gradient = Gradient.subs(zip(variables,x))
    Gradient = np.array(Gradient).astype(np.float64)

    return Gradient


def get_hessian(function, x):
    variables = list(ordered(function.free_symbols))
    Hessian = simplify(derive_by_array(derive_by_array(function, variables), variables))
    Hessian = Hessian.subs(zip(variables,x))
    Hessian = np.array(Hessian).astype(np.float64)

    return Hessian

def update_qcf(obj_function,eq_const,my):
    qfc = obj_function
    for i in eq_const:
        qfc += 1/(2*my)*i**2
    return qfc



def quadratic_penalty_method(obj_function,eq_const,steps,my_start,x0,epsilion):
    tau = tau0
    my = my_start
    x = x0
    for i in range(steps):
        qcf = update_qcf(obj_function,eq_const,my)
        x = newton(qcf, x, tau)
        my = my*0.1
    return x


x1,x2 = symbols('x1,x2')


my = 10
epsilon = 0.1
tau0 = 0.1
x0 = [0,0]
steps = 10

#object function
obj_func  = x1 + x2

#equality constrains
eq_const = []
eq_const.append(x1**2 + x2**2 -2) 
eq_const.append(x1 -1 -x2)


x = quadratic_penalty_method(obj_func,eq_const,10,my,x0,0.1)
print(x)

# qcf = update_qcf(obj_func,eq_const,my);

# x = x0
# x = newton(qcf,x,tau0)

# my = my*0.1

# for i in range(steps):
#     qcf = update_qcf(obj_func,eq_const,my);
#     x = newton(qcf,x,tau0)
#     my = my*0.1
# print(x)
