# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:18:34 2016

@author: jake
"""


from dolfin import *
import numpy as np

mesh = UnitSquareMesh(3, 3)
#V = FunctionSpace(mesh, "CR", 1)

element = FiniteElement("Discontinuous Lagrange Trace", "triangle", 0)

V1 = FunctionSpace(mesh, FiniteElement("Discontinuous Lagrange Trace", "triangle", 0))

#print V.dim()
print V1.dim()

#f = Function(V1)
f = interpolate(Constant(1.0), V1)
#plot(f, interactive = True)
#File('f.pvd') << f

v = TestFunction(V1)
print assemble((f * v)('+') * dS).array()
print assemble((f * v)('-') * dS).array()



plot(mesh, interactive = True)
quit()

# Unknown
u = project(Constant(1.0), V)
# Solution at last time step
u_prev = Function(V)
u_prev.assign(u)

v = TestFunction(V)
dt = Constant(0.001)

# Set up variational form for Netwon's method
du = TrialFunction(V)
a = avg((du - u**(-1.0/2.0) * du * dt) * v) * dS
L = -avg(((u - u_prev) - sqrt(u) * dt) * v) * dS

print assemble(a).array()

quit()

# Setup algorithm for Newton's method
du = Function(V)
u  = Function(V)
omega = 1.0       # relaxation parameter
eps = 1.0
tol = 1.0E-5
i = 0
maxiter = 25
while eps > tol and i < maxiter:
    i += 1
    A, b = assemble_system(a, L)
    solve(A, du.vector(), b)
    eps = np.linalg.norm(du.vector().array(), ord=np.Inf)
    print 'Norm:', eps
    u.vector()[:] = u_prev.vector() + omega*du.vector()
    u_prev.assign(u)
    
    #plot(u, interactive = True)
    print du.vector().array()
    print u.vector().array()

quit()


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
