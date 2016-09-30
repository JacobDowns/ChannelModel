# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:18:34 2016

@author: jake
"""


from dolfin import *
import numpy as np

mesh = UnitSquareMesh(2, 2)

# Trace test function 
V_tr = FunctionSpace(mesh, FiniteElement("Discontinuous Lagrange Trace", "triangle", 0))

# Unknown
u = interpolate(Constant(1.0), V_tr)
# Unknown at previous time step
u_prev = Function(V_tr)
u_prev.assign(u)
# Trace test function
v = TestFunction(V_tr)

# Variational form for ODE
dt = Constant(0.001)
F = (((u - u_prev) - sqrt(u) * dt) * v)('+') * dS + u*v * ds

# Jacobian
du = TrialFunction(V_tr)
J = derivative(F, u, du) 

print assemble(J).array()
print
print assemble(F).array()

# Solve ODE
t = 0.0
ts = []
ys = []

while t < 0.5:    
    print t
    solve(F == 0, u, J = J)
    print u.vector().array()
    quit()
    u_prev.assign(u)
    ts.append(t)
    ys.append(u.vector().array()[1])
    t += dt.values()[0]

print u.vector().array() 

from pylab import *

# Compare to analytic solution
ts = array(ts)
#plot(ts, ys)
plot(ts, 0.25*(ts + 2.0)**2 - ys, 'k')
show()

