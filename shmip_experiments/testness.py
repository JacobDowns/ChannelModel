from dolfin import *

mesh = UnitSquareMesh(3,3)
V = FunctionSpace(mesh, 'CG', 1)
f = project(Expression('x[0]'),V)
