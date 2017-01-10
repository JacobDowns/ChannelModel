# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:18:34 2016

@author: jake
"""


from dolfin import *
import numpy as np

parameters['ghost_mode'] = 'shared_facet'

# Create mesh
mesh = UnitSquareMesh(32, 32)

# Define function spaces and mixed (product) space
V_cg = FunctionSpace(mesh, "CG", 1)



u_prev = project(Constant(1.0), V_cg)
v = TestFunction(V_cg)
dt = Constant(0.0001)
u = Function(V_cg)
u.assign(u_prev)

#plot(u, interactive = True)

#F = avg(((u - u_prev) - abs(sqrt(u + Constant(1e-10))) * dt) * v) * dS

#F = avg(((u - u_prev) - sqrt(u) * dt) * v) * dS
F = (((u - u_prev) - sqrt(u) * dt) * v) * dx


du = TrialFunction(V_cg)
J = derivative(F, u, du) 

utrial = TrialFunction(V_cg)
F1 = (utrial - u**(-1.0/2.0) * utrial * dt) * v * dx



A = assemble(F1).array()
B = assemble(J).array()
C = A - B
print C.min()
print C.max()
quit()


#print assemble(avg(u**(1.0 / 2.0)) * dS).array()
#quit()


"""
f = FacetFunctionDouble(mesh)

midpoints = np.zeros((V_cg.dim(), 2))
i = 0
for f in facets(mesh):
  print f.midpoint()[0]
  x = f.midpoint()[0]
  y = f.midpoint()[1]
  midpoints[i][0] = x
  midpoints[i][1] = y
  i += 1
  
print midpoints
  
quit()"""
#a = lhs(F)
#L = rhs(F)


#u = Function(V_cg)

time = 0.0
print time

ts = []
ys = []
while time < 0.5:    
    solve(F == 0, u, J = J)
    u_prev.assign(u)
    ts.append(time)
    ys.append(u.vector().array()[0])
    time += 0.001
    
#plot(u, interactive = True)

from pylab import *
  
plot(ts, ys)
#plot(ts, 1.0 / (1.0 - array(ts)), 'k')
show()