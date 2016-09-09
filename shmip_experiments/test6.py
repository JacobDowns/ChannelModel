from dolfin import *
from dolfin import MPI, mpi_comm_world
import numpy as np 

#parameters['ghost_mode'] = 'shared_facet'
parameters['ghost_mode'] = 'shared_facet'

# Process number
MPI_rank = MPI.rank(mpi_comm_world())

mesh = UnitSquareMesh(3,3)
#plot(mesh, interactive = True)

a = FacetFunction('size_t', mesh)

#a.array()[:] = np.array(range(len(a.array())))
File('a.xml') >> a
print a.array()
quit()

V_cg = FunctionSpace(mesh, 'Lagrange', 1) 
V_cr = FunctionSpace(mesh, 'CR', 1)

f = Function(V_cg)
g = Function(V_cr)
File('f.xml') >> f
File('g.xml') >> g

v = TestFunction(V_cg)

L = len(f.vector().array())
c = V_cg.tabulate_dof_coordinates().reshape(L, 2)
a = f.vector().array()
b = assemble(v * dx)

x = Vector()
y = Vector()
b.gather(x, np.array(range(V_cg.dim()), "intc"))
f.vector().gather(y, np.array(range(V_cg.dim()), "intc"))

if MPI_rank == 0:
  indexes = np.argsort(y.array())
  print y.array()[indexes]
  print x.array()[indexes]
quit()

A = np.zeros((L, 4))
A[:,0] = a
A[:,1] = c[:,0]
A[:,2] = c[:,1]
A[:,3] = b.array()

print A
