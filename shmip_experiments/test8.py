# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:37:36 2016

@author: jake
"""
from dolfin import *
import numpy as np

mesh = UnitSquareMesh(3,3)


V1 = FunctionSpace(mesh, 'CG', 1) 
V2 = FunctionSpace(mesh, 'CG', 1) 
V3 = FunctionSpace(mesh, 'CR', 1)
V = V1 * V2 * V3

(v1, v2) = TestFunctions(V)
print v1[0]
print v1[1]
print v2
quit()

"""
h = Function(V1)
h.vector()[:] = np.array(range(V1.dim()))
h1 = interpolate(h, V3)

print h1.vector().array()

print h.vector().array()

v = TestFunction(V3)


print assemble((h * v)('+') * dS).array()
print assemble((h1 * v)('+') * dS).array()
quit()"""

f = Function(V)

(f1,f3) = f.split(True)
(f1,f2) = f1.split(True)

a = Function(V1)
b = Function(V2)
c = Function(V3)

a.vector()[:] = np.array(range(V1.dim()))
b.vector()[:] = np.array(range(V2.dim()))
c.vector()[:] = np.array(range(V3.dim()))

f1.assign(a)
f2.assign(b)
f3.assign(c)


phi = f.sub(0).sub(0)
h = f.sub(0).sub(1)
S = f.sub(1)

assign(phi, a)
assign(h, b)
assign(S, c)

print V.tabulate_dof_coordinates()

print f.vector().array()
#print V_cg.dim()
#print V_cr.dim()