import sympy
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot, plot3d, PlotGrid
from sympy import solve, Poly, Eq, Function, exp
from sympy import symbols, Matrix, Transpose
from sympy import *
from sympy.vector import CoordSys3D, gradient
from mpl_toolkits.mplot3d import Axes3D
from sympy import lambdify

#variables = list(ordered(f.free_symbols))
#Gradient = simplify(derive_by_array(f, v))
#Hessian = simplify(derive_by_array(derive_by_array(function, variables), variables))
#Gradient.subs(zip(v,x))
#test = np.array(Gradient).astype(np.float64

def get_gradient_symb(function):
    variables = list(ordered(function.free_symbols))
    Gradient = simplify(derive_by_array(function, variables))

    return Gradient

def get_hessian_symb(function, dim):
    variables = list(ordered(function.free_symbols))
    Hessian = simplify(derive_by_array(derive_by_array(function, variables), variables))

    return hessian


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
    qfc = obj_fucntion
    for i in eq_const:
        qfc += 1/(2*my)*i**2
    return qfc

# basic newton algorithm 
def newton(function,x0,epsilon):
    grad = get_gradient(function,x0)
    hess = get_hessian(function,x0)
    d = -1*inv(hess).dot(grad)
    x = x0 + d

    while np.abs(norm(x) - norm(x_prev)) >= eps:
        grad = get_gradient(function,x)
        hess = get_hessian(function,x)
        d = -1*inv(hess).dot(grad)
        x = x0 + d

    return x