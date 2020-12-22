import sympy
import numpy as np
import matplotlib.pyplot as plt
from sympy.plotting import plot, plot3d, PlotGrid
from sympy import symbols, Matrix, Transpose
from sympy.vector import CoordSys3D, gradient
from mpl_toolkits.mplot3d import Axes3D
from sympy import lambdify


x,y = sympy.symbols('x,y')
rosenbrock = (1-x)**2 + 100*(y-x**2)**2

lamb_rosenbrock = lambdify([x,y],rosenbrock,'numpy')

x1, x2 = symbols('x1 x2')

#Startpunkt
x1 = []
x2 = []
value = []

x1.append(-2)
x2.append(-1)
value.append(rosenbrock.subs({x:x1[0], y: x2[0]}))


steps = 100;
for i in range(steps):
    gradx = rosenbrock.diff(x)
    gradx_value = gradx.subs({x:x1[i], y: x2[i]})
    grady = rosenbrock.diff(y)
    grady_value = gradx.subs({x:x1[i], y: x2[i]})

    # print('gradx_value',gradx_value)
    # print('grady_value',grady_value)


    x1.append(x1[i]-0.001*gradx_value) 
    x2.append(x2[i]-0.001*grady_value) 

    value.append(rosenbrock.subs({x:x1[i], y: x2[i]}))
    #print(value[i])

fig = plt.figure()
ax = plt.axes(projection='3d')


# lam_rosenbrock = rosenbrock.lambdify([(x,y)])
#lamb_rosenbrock = lambdify([x,y],rosenbrock,'numpy')
x = np.linspace(-3, 3, 30)
y = np.linspace(-3, 3, 30) 
X, Y = np.meshgrid(x, y)

Z = lamb_rosenbrock(X,Y)

        
for i in range(steps):
    ax.scatter(x1[i],x2[i],value[i],color='red')


ax.contour3D(X, Y, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none');

plt.show()