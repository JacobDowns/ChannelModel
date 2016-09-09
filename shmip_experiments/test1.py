from dolfin import *
from dolfin import MPI, mpi_comm_world
import numpy as np 

parameters['ghost_mode'] = 'shared_vertex'

# Process number
MPI_rank = MPI.rank(mpi_comm_world())

mesh = UnitSquareMesh(3,3)
plot(mesh, interactive = True)

V_cg = FunctionSpace(mesh, 'Lagrange', 1) 
V_cr = FunctionSpace(mesh, 'CR', 1)

f = Function(V_cg)
g = Function(V_cr)
#f.vector()[:] = np.array(range(V_cg.dim()))
#g.vector()[:] = np.array(range(V_cr.dim()))
File('f.xml') >> f
File('g.xml') >> g

v = TestFunction(V_cg)

# Normal and tangent vectors 
n = FacetNormal(mesh)
t = as_vector([n[1], -n[0]])
dv_ds = dot(grad(v), t)

L = len(f.vector().array())
c = V_cg.tabulate_dof_coordinates().reshape(L, 2)
a = f.vector().array()
b = assemble(dv_ds('+') * dS).array()

A = np.zeros((L, 4))
A[:,0] = a
A[:,1] = c[:,0]
A[:,2] = c[:,1]
A[:,3] = b

print A
