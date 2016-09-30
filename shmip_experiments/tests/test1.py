# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:18:34 2016

@author: jake
"""


from dolfin import *

mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "CG", 1)

# Unknown
u = project(Constant(1.0), V)
# Solution at last time step
u_prev = Function(V)
u_prev.assign(u)

v = TestFunction(V)
dt = Constant(0.001)

# Variational form for ODE du/dt = sqrt(u), u(0) = 1
F = (((u - u_prev) - sqrt(u) * dt) * v) * dx

du = TrialFunction(V)
J = derivative(F, u, du) 

# Manually define Jacobian to compare to the one fenics computes 
#u_trial = TrialFunction(V)
#J_manual = (u_trial - u**(-1.0/2.0) * u_trial * dt) * v * dx

# Solve ODE
t = 0.0
ts = []
ys = []
while t < 0.5:    
    solve(F == 0, u, J = J)
    u_prev.assign(u)
    ts.append(t)
    ys.append(u.vector().array()[0])
    t += dt.values()[0]
    

from pylab import *

# Compare to analytic solution
ts = array(ts)
plot(ts, ys)
plot(ts, 0.25*(ts + 2.0)**2, 'k')
show()
