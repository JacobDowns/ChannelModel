# -*- coding: utf-8 -*-
"""
Test edge integrals for Discontinuous Lagrance Trace finite elements.
"""


from dolfin import *
import numpy as np

mesh = UnitSquareMesh(2, 2)

# Discontinuous trace element space
V_tr = FunctionSpace(mesh, FiniteElement("Discontinuous Lagrange Trace", "triangle", 0))
# CG  element space
V_cg = FunctionSpace(mesh, 'CG', 1)
# CR element space
V_cr = FunctionSpace(mesh, 'CR', 1)

# CG funcgion
f = interpolate(Expression("x[0]"), V_cg) #Function(V_cg)
# Normal and tangent vectors 
n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])
# Edge derivative
df_ds = dot(grad(f), t)

# Trace element test function
v_tr = TestFunction(V_tr)
# Usual CR element test function
v_cr = TestFunction(V_cr)

print assemble((df_ds * v_tr)('+') * dS).array()
print assemble((df_ds * v_tr)('-') * dS).array()
print
print assemble((df_ds * v_cr)('+') * dS).array()
print assemble((df_ds * v_cr)('-') * dS).array()

