%matplotlib inline
import sympy
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
from sympy_lib import *

x1,x2 = symbols('x1,x2')

#objective function
obj_func = (1-x1)**2 + 100*(x2-x1**2)**2


def steepest_descent(function,x0,epsilion):
    grad = get_gradient(function,x0)
    x = x0 - 0.01*grad
    while np.abs(norm(x) - norm(x_prev)) >= eps:
        grad = get_gradient(function,x0)
        x = x0 - 0.01*grad

    return x


eps = 0.2
x0 = [0,0]
steepest_descent(obj_func,x0,eps)