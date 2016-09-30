# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:18:34 2016

@author: jake
"""


from dolfin import *
import numpy as np

mesh = UnitSquareMesh(2, 2)
plot(mesh, interactive = True)

V = FunctionSpace(mesh, FiniteElement("Discontinuous Lagrange Trace", "triangle", 0))

# Unknown
u = interpolate(Constant(1.0), V)
# Unknown at previous time step
u_prev = Function(V)
u_prev.assign(u)

v = TestFunction(V)

print assemble((u * v)('+') * dS).array()
print assemble((u * v)('-') * dS).array()

mask = assemble(v * ds).array()

V_cg = FunctionSpace(mesh, 'CG', 1)
f  = interpolate(Expression("x[0]"), V_cg)
print assemble((f * v)('+') * dS).array()

print mask
quit()

dt = Constant(0.001)


F = (((u - u_prev) - sqrt(u) * dt) * v)('+') * dS + u*v * ds

print assemble(F).array()

du = TrialFunction(V)
J = derivative(F, u, du) 


A = assemble(J).array()
np.savetxt('out.txt', A, newline = '\n\n')

solve(F == 0, u, J = J)

